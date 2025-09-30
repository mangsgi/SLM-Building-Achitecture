# tasks/train.py  (step-logging + notify + cooperative stop + continuous status SSE)
import os
import json
import socket
import time
from pathlib import Path

import mlflow
import torch
from celery_app import celery_app
from celery.utils.log import get_task_logger
from mlflow.tracking import MlflowClient

from ml.models.factory import build_model_from_json
from .dataset import create_dataloader_v1, load_training_data
from .tokenizers import choose_tokenizer_from_config

# Redis Pub/Sub (SSE 중계)
from deps import rds_sync, train_event_channel

# (옵션) 학습 완료 시 FastAPI로 알림 POST
import requests  # noqa: F401

logger = get_task_logger(__name__)

# ===================== DeepSpeed 관련 함수 =====================
def _is_rank0(engine_or_none):
    return (engine_or_none is None) or (getattr(engine_or_none, "global_rank", 0) == 0)

def _maybe_import_deepspeed(use_deepspeed: bool):
    if not use_deepspeed:
        return None
    import os, torch
    os.environ.setdefault("DS_BUILD_OPS", "0")
    os.environ.setdefault("DS_SKIP_CUDA_CHECK", "1")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU가 없는데 use_deepspeed=True 입니다.")
    import deepspeed
    return deepspeed

# ===================== 유틸 =====================
def _as_int(x, default):
    if x is None:
        return int(default)
    try:
        return int(x)
    except Exception:
        return int(default)

def _as_str(x, default):
    return str(x) if x not in (None, "") else str(default)

def _looks_http(uri: str) -> bool:
    return isinstance(uri, str) and uri.startswith(("http://", "https://"))

def _can_connect_http(uri: str, timeout=1.5) -> bool:
    """간단 소켓 체크로 MLflow 서버 연결성 확인 (파일 백엔드는 True)"""
    try:
        from urllib.parse import urlparse
        u = urlparse(uri)
        if u.scheme not in ("http", "https"):
            return True
        if not u.hostname:
            return False
        port = u.port or (443 if u.scheme == "https" else 80)
        with socket.create_connection((u.hostname, port), timeout=timeout):
            return True
    except Exception:
        return False

def _setup_mlflow_tracking():
    """MLFLOW_TRACKING_URI 적용 + 타임아웃 설정. HTTP 불가시 경고만 남기고 진행."""
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "5")  # 초
    uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    mlflow.set_tracking_uri(uri)
    if _looks_http(uri) and not _can_connect_http(uri):
        logger.warning("[MLflow] '%s' 접속 불가. run_id로 start_run 할 때 실패할 수 있습니다.", uri)
    logger.info("MLflow Tracking URI = %s", mlflow.get_tracking_uri())

def publish_event(task_id: str, event: str, data: dict):
    """학습 상태를 Redis Pub/Sub으로 전파 (실패해도 학습은 계속)"""
    try:
        payload = {"event": event, "data": data}
        rds_sync.publish(train_event_channel(task_id), json.dumps(payload, ensure_ascii=False))
    except Exception as e:
        logger.warning(f"[events] publish failed: {e}")

def publish_status(task_id: str, state: str, **extra):
    """일관된 status 이벤트 (running/finished/stopped/error)"""
    payload = {"task_id": task_id, "state": state}
    if extra:
        payload.update(extra)
    publish_event(task_id, "status", payload)

# ---------- STOP FLAG helpers (reason 제거) ----------
def _stop_key(task_id: str) -> str:
    return f"train:stop:{task_id}"

def _clear_stop_flag(task_id: str):
    try:
        rds_sync.delete(_stop_key(task_id))
    except Exception:
        pass

def _should_stop(task_id: str) -> bool:
    try:
        flag = rds_sync.get(_stop_key(task_id))
        return flag in ("1", b"1")
    except Exception:
        return False

# ===================== 손실/평가 =====================

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)  # [B, T, V]
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1),      # [B*T, V]
        target_batch.flatten()     # [B*T]
    )
    return loss

def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.0
    if len(data_loader) == 0:
        return None
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i >= num_batches:
            break
        loss = calc_loss_batch(input_batch, target_batch, model, device)
        total_loss += loss.item()
    return total_loss / max(1, num_batches)

def evaluate_model(model, train_loader, val_loader, device, eval_iter=10):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss

# ===================== 메인 학습 태스크 =====================

@celery_app.task(bind=True)
def train_and_infer_from_json(self, request_json: dict):
    # ---------- 1) 파라미터 파싱 ----------
    config         = request_json.get("config", {}) or {}
    layer_json     = request_json.get("model", []) or []
    dataset_name   = request_json.get("dataset", "tiny_shakespeare")
    dataset_config = request_json.get("dataset_config", "default")
    model_name     = request_json.get("modelName", "trained_model")
    run_id         = request_json.get("run_id")  # 라우터에서 선생성
    task_id        = self.request.id

    batch_size     = _as_int(config.get("batch_size"), 4)
    epochs         = _as_int(config.get("epochs"), 5)
    seq_max_length = _as_int(config.get("context_length"), 32)
    stride         = _as_int(config.get("stride"), seq_max_length)
    dtype          = _as_str(config.get("dtype"), "fp32")

    # step 로깅 주기 / SSE 스텝 이벤트 토글 / 완료 알림 URL
    log_every = _as_int(os.getenv("LOG_EVERY_STEPS"), 1)
    enable_sse_step = os.getenv("ENABLE_SSE_STEP_EVENTS", "0") == "1"
    notify_url = os.getenv("BACKEND_NOTIFY_URL")

    logger.info(
        f"[TASK] start | task_id={task_id}, model_name={model_name}, run_id={run_id}, "
        f"epochs={epochs}, batch_size={batch_size}, dtype={dtype}, "
        f"context_length={seq_max_length}, stride={stride}, log_every={log_every}"
    )
    self.update_state(state="STARTED")

    # run_id 필수 확인
    if not run_id:
        msg = "run_id missing (router must create MLflow run)."
        logger.error(msg)
        publish_event(task_id, "error", {"task_id": task_id, "message": msg})
        publish_status(task_id, "error", message=msg)
        return {"status": "error", "message": msg}

    # ---------- STOP flag 초기화 ----------
    _clear_stop_flag(task_id)

    # ---------- 2) MLflow 트래킹 설정 ----------
    _setup_mlflow_tracking()

    try:
        # ---------- 3) 토크나이저/모델 준비 ----------
        tokenizer = choose_tokenizer_from_config(config)
        logger.info(f"Tokenizer ready ({config.get('model','gpt-2')}), "
                    f"vocab={getattr(tokenizer, 'n_vocab', 'N/A')}")

        if not isinstance(layer_json, list):
            raise ValueError("layer_json must be a list of layer configurations")

        model = build_model_from_json(layer_json, dtype=dtype)
        total_params     = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model built. total={total_params:,}, trainable={trainable_params:,}")

        # [추가] LM Head 유효성 검사: GPU로 보내기 전에 차원 검사
        vocab_size = getattr(tokenizer, 'n_vocab', None)
        if vocab_size is None and hasattr(tokenizer, 'vocab_size'):
            vocab_size = tokenizer.vocab_size

        if vocab_size and hasattr(model, 'layers') and model.layers:
            last_layer = model.layers[-1]

            if not isinstance(last_layer, torch.nn.Linear) or last_layer.out_features != vocab_size:
                raise ValueError(
                    f"Model's last layer is not a valid LM Head for vocab size {vocab_size}. "
                    f"Expected nn.Linear(out_features={vocab_size}), but found {type(last_layer)} "
                    f"with out_features={getattr(last_layer, 'out_features', 'N/A')}."
                )


        # ---------- 4) 데이터 로드/분할 ----------
        logger.info(f"Loading dataset: {dataset_name}/{dataset_config}")
        if dataset_name == "allenai/c4" or dataset_name == "roneneldan/TinyStories" or dataset_name == "mychen76/openwebtext-100k":
            training_text = load_training_data(dataset_name, dataset_config,
                                   split="train", 
                                   streaming=True, max_rows=50000)
        else:
            training_text = load_training_data(dataset_name, dataset_config)
        training_text = training_text[:]  # 샘플 컷 (원하면 제거)
        split_idx = int(len(training_text) * 0.8)
        train_text, val_text = training_text[:split_idx], training_text[split_idx:]

        # ---------- 5) 데이터로더 ----------
        train_loader = create_dataloader_v1(
            txt=train_text, batch_size=batch_size,
            max_length=seq_max_length, stride=stride,
            shuffle=True, num_workers=0, tokenizer=tokenizer,
        )
        val_loader = create_dataloader_v1(
            txt=val_text, batch_size=batch_size,
            max_length=seq_max_length, stride=stride,
            shuffle=False, num_workers=0, tokenizer=tokenizer,
        )
        logger.info(f"Dataloaders ready. train_batches={len(train_loader)}, val_batches={len(val_loader)}")

        # ---------- 6) 장비/옵티마이저 ----------
        use_deepspeed = False 
        deepspeed = _maybe_import_deepspeed(use_deepspeed)
        ds_engine = None

        if use_deepspeed:
            assert deepspeed is not None, "DeepSpeed not installed"
            
            # 백엔드 기본값(없으면) + 프론트 오버라이드
            ds_cfg = json.load(open("tasks/files/deepspeed_config.json"))
            # DeepSpeed가 optimizer를 만들지 않으면, 기존 optimizer를 넘길 수도 있습니다.
            # 여기선 DS가 직접 생성(권장): model_parameters만 넘김
            ds_engine, optimizer, _, _ = deepspeed.initialize(
                model=model,
                model_parameters=(p for p in model.parameters() if p.requires_grad),
                config=ds_cfg,
            )
            model = ds_engine           # 이후 학습에서 model로 사용
            device = ds_engine.device
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)

        eval_freq, eval_iter = 1, 5
        global_step = 0
        last_train_loss = None
        last_val_loss = None

        logger.info(f"Training on device: {device}")

        # ---------- 7) MLflow run에 붙기 + started 이벤트 ----------
        mlflow_url = None
        exp_id = None
        try:
            client = MlflowClient()
            run_info = client.get_run(run_id).info
            exp_id = run_info.experiment_id
            tracking_uri = mlflow.get_tracking_uri().rstrip("/")
            mlflow_url = f"{tracking_uri}/#/experiments/{exp_id}/runs/{run_id}"
            logger.warning(f"🏃 View run at: {mlflow_url}")
            logger.warning(f"🧪 View experiment at: {tracking_uri}/#/experiments/{exp_id}")
        except Exception as e:
            logger.warning("[MLflow] get_run 실패: %s", e)

        # 시작 알림
        publish_event(task_id, "started", {
            "task_id": task_id,
            "model_name": model_name,
            "run_id": run_id,
            "mlflow_url": mlflow_url,
            "mlflow_experiment_id": exp_id,
            "epochs": epochs,
            "batch_size": batch_size,
            "context_length": seq_max_length,
            "stride": stride
        })
        publish_status(task_id, "running", epoch=0, epochs=epochs, global_step=0)

        # ---------- 8) 학습 + (스텝/에포크) 로깅 ----------
        try:
            # 중첩 run 방지
            if mlflow.active_run():
                mlflow.end_run()

            ema = None
            ema_alpha = float(os.getenv("STEP_LOSS_EMA_ALPHA", "0.1"))

            with mlflow.start_run(run_id=run_id):
                for epoch in range(epochs):
                    model.train()
                    epoch_loss, batch_count = 0.0, 0

                    for step_idx, (input_batch, target_batch) in enumerate(train_loader, start=1):
                        # 협력적 중단: 매 스텝 직전에 플래그 확인
                        if _should_stop(task_id):
                            # 학습 중단 처리
                            try:
                                mlflow.set_tag("training_status", "stopped_by_user")
                                mlflow.log_metric("stopped_at_global_step", global_step)
                                mlflow.log_metric("stopped_at_epoch", epoch + 1)
                            except Exception:
                                pass

                            publish_event(task_id, "stopped", {
                                "task_id": task_id,
                                "global_step": global_step,
                                "epoch": epoch + 1,
                                "status": "stopped"
                            })
                            publish_status(task_id, "stopped", epoch=epoch+1, global_step=global_step)

                            # 리소스 정리
                            try:
                                del train_loader, val_loader, optimizer
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                            except Exception:
                                pass

                            return {
                                "status": "stopped",
                                "message": "Training stopped by request.",
                                "stopped_at_step": global_step,
                                "stopped_at_epoch": epoch + 1
                            }

                        if ds_engine is not None:
                            model.zero_grad()
                            loss = calc_loss_batch(input_batch, target_batch, model, device)
                            model.backward(loss)   # ds_engine.backward
                            model.step()           # ds_engine.step (grad accumulation은 ds_config가 반영)
                        else:
                            optimizer.zero_grad()
                            loss = calc_loss_batch(input_batch, target_batch, model, device)
                            loss.backward()
                            optimizer.step()

                        # ---------- 스텝 단위 ----------
                        global_step += 1
                        epoch_loss  += loss.item()
                        batch_count += 1

                        # EMA 갱신
                        ema = loss.item() if ema is None else (ema_alpha * loss.item() + (1 - ema_alpha) * ema)

                        # 주기적 로깅
                        if (global_step % log_every) == 0:
                            mlflow.log_metric("train/step_loss", loss.item(), step=global_step)
                            mlflow.log_metric("train/step_loss_ema", ema, step=global_step)
                            logger.info(f"[step {global_step}] loss={loss.item():.4f} | ema={ema:.4f}")

                            if enable_sse_step:
                                publish_event(task_id, "step", {
                                    "task_id": task_id,
                                    "global_step": global_step,
                                    "epoch": epoch + 1,
                                    "loss": float(loss.item()),
                                    "ema_loss": float(ema),
                                })
                                # 상태도 같이 한 번 더
                                publish_status(task_id, "running", epoch=epoch+1, global_step=global_step)

                    # ---------- 에포크 단위 ----------
                    avg_epoch = epoch_loss / max(batch_count, 1)
                    mlflow.log_metric("train/epoch_loss", avg_epoch, step=epoch)
                    logger.info(f"Epoch {epoch+1}/{epochs} | avg_loss={avg_epoch:.4f}")

                    # 정기 평가 + 진행 이벤트
                    last_train_loss, last_val_loss = evaluate_model(
                        model, train_loader, val_loader, device, eval_iter=eval_iter
                    )
                    if last_train_loss is not None:
                        mlflow.log_metric("eval/train_loss", last_train_loss, step=epoch)
                    if last_val_loss is not None:
                        mlflow.log_metric("eval/val_loss", last_val_loss, step=epoch)

                    publish_event(task_id, "progress", {
                        "task_id": task_id,
                        "epoch": epoch + 1,
                        "epochs": epochs,
                        "avg_epoch_loss": avg_epoch,
                        "train_loss": last_train_loss,
                        "val_loss": last_val_loss,
                        "global_step": global_step,
                        "run_id": run_id
                    })
                    # 진행 상태 갱신 (에포크마다 최소 1회)
                    publish_status(task_id, "running", epoch=epoch+1, epochs=epochs, global_step=global_step)

                # ---------- 9) 학습 결과 저장 ----------
                COMPLETED_DIR = Path("completed")
                COMPLETED_DIR.mkdir(exist_ok=True)
                save_path = COMPLETED_DIR / f"{model_name}.pt"

                bundle = {
                    "layers": layer_json,
                    "config": {
                        **config,
                        "dtype": dtype,
                        "context_length": seq_max_length,
                    },
                    "state_dict": model.state_dict(),
                }
                
                if ds_engine is not None:
                    if _is_rank0(ds_engine):
                        bundle["state_dict"] = ds_engine.module.state_dict()
                        torch.save(bundle, save_path)
                    if ds_engine is not None:
                        ds_engine.barrier()  # 모든 랭크 동기화
                else:
                    torch.save(bundle, save_path)

                # 아티팩트 기록 (실패해도 학습은 완료로 간주)
                try:
                    mlflow.log_artifact(str(save_path))
                except Exception as e:
                    logger.warning("[MLflow] artifact log skipped: %s", e)

        except Exception:
            # 필요시 no-mlflow 경로로 전환 가능. 지금은 에러 전파.
            raise

        # ---------- 10) 완료 알림 ----------
        finished_payload = {
            "task_id": task_id,
            "model_name": model_name,
            "run_id": run_id,
            "completed_model_path": str(save_path),
            "last_train_loss": last_train_loss,
            "last_val_loss": last_val_loss,
            "status": "finished",
            "ts": time.time(),
        }

        publish_event(task_id, "finished", finished_payload)
        publish_status(task_id, "finished")

        # (옵션) FastAPI 라우트로 POST
        if notify_url:
            try:
                resp = requests.post(notify_url, json=finished_payload, timeout=5)
                resp.raise_for_status()
                logger.info(f"[Notify] POST 성공 → {notify_url}")
            except Exception as e:
                logger.warning(f"[Notify] POST 실패: {e}")

        return {
            "status": "success",
            "message": "Training complete",
            "completed_model_path": str(save_path),
        }

    except Exception as e:
        logger.exception("Training failed")
        err_payload = {
            "task_id": task_id,
            "model_name": model_name if 'model_name' in locals() else None,
            "run_id": run_id if 'run_id' in locals() else None,
            "message": str(e)
        }
        publish_event(task_id, "error", err_payload)
        publish_status(task_id, "error", message=str(e))
        raise
