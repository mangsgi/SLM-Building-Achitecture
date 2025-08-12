import mlflow
from celery_app import celery_app
from celery.utils.log import get_task_logger
from ml.models.factory import build_model_from_json
import torch
from .dataset import create_dataloader_v1, load_training_data
from .tokenizers import choose_tokenizer_from_config
import os

logger = get_task_logger(__name__)

# ====== 평가/손실 계산 ======
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss

def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.0
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches

def evaluate_model(model, train_loader, val_loader, device, eval_iter=10):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


# ====== 메인 학습 태스크 ======
@celery_app.task(bind=True)
def train_and_infer_from_json(self, request_json: dict):
    # 1) 파라미터 파싱
    config         = request_json.get("config", {})
    layer_json     = request_json.get("model", [])
    dataset_name   = request_json.get("dataset", "tiny_shakespeare")
    dataset_config = request_json.get("dataset_config", "default")
    model_name     = request_json.get("modelName", "trained_model")
    run_id         = request_json.get("run_id")

    # 학습/모델 관련 파라미터
    batch_size  = int(config.get("batch_size", 4))
    epochs      = int(config.get("epochs", 5))
    dtype       = config.get("dtype", "fp32")

    # ✅ context_length/stride 적용
    seq_max_length = int(config.get("context_length", 32))
    stride = int(config.get("stride", max(1, seq_max_length // 2)))

    logger.info(
        f"[TASK] start | model_name={model_name}, run_id={run_id}, "
        f"epochs={epochs}, batch_size={batch_size}, dtype={dtype}, "
        f"context_length={seq_max_length}, stride={stride}"
    )
    self.update_state(state="STARTED")

    # 2) run_id 필수
    if not run_id:
        logger.error("run_id missing (router must create MLflow run).")
        return {"status": "error", "message": "run_id missing (router must create MLflow run)."}

    # 3) MLflow 트래킹 설정
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    try:
        # 4) 토크나이저/모델 준비
        tokenizer = choose_tokenizer_from_config(config)
        logger.info(f"Tokenizer ready ({config.get('model','gpt-2')}), vocab={getattr(tokenizer,'n_vocab','N/A')}")

        if not isinstance(layer_json, list):
            raise ValueError("layer_json must be a list of layer configurations")

        model = build_model_from_json(layer_json, dtype=dtype)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model built. total={total_params:,}, trainable={trainable_params:,}")

        # (선택) positionalEmbedding.ctxLength와 config.context_length 불일치 경고
        try:
            pe_ctx = next(
                (n.get("data", {}).get("ctxLength")
                 for n in layer_json if n.get("type") == "positionalEmbedding"),
                None
            )
            if pe_ctx is not None and int(pe_ctx) != seq_max_length:
                logger.warning(
                    f"[WARN] config.context_length({seq_max_length}) != positionalEmbedding.ctxLength({pe_ctx}). "
                    "학습은 config 값을 사용합니다."
                )
        except Exception:
            pass

        # 5) 데이터 로드/분할
        logger.info(f"Loading dataset: {dataset_name}/{dataset_config}")
        training_text = load_training_data(dataset_name, dataset_config)
        # 과도한 길이 방지 (원하면 조절)
        training_text = training_text[:20000]
        split_idx = int(len(training_text) * 0.8)
        train_text, val_text = training_text[:split_idx], training_text[split_idx:]

        # 6) 데이터로더
        train_loader = create_dataloader_v1(
            txt=train_text,
            batch_size=batch_size,
            max_length=seq_max_length,
            stride=stride,
            shuffle=True,
            num_workers=0,
            tokenizer=tokenizer,
        )
        val_loader = create_dataloader_v1(
            txt=val_text,
            batch_size=batch_size,
            max_length=seq_max_length,
            stride=stride,
            shuffle=False,
            num_workers=0,
            tokenizer=tokenizer,
        )
        logger.info(f"Dataloaders ready. train_batches={len(train_loader)}, val_batches={len(val_loader)}")

        # 7) 장비/옵티마이저
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
        eval_freq, eval_iter = 2, 5
        train_losses, val_losses, global_step = [], [], 0
        logger.info(f"Training on device: {device}")

        # 8) 학습 + MLflow 로깅
        with mlflow.start_run(run_id=run_id):
            for epoch in range(epochs):
                model.train()
                epoch_loss, batch_count = 0.0, 0

                for input_batch, target_batch in train_loader:
                    optimizer.zero_grad()
                    loss = calc_loss_batch(input_batch, target_batch, model, device)
                    loss.backward()
                    optimizer.step()

                    global_step += 1
                    epoch_loss  += loss.item()
                    batch_count += 1

                    if global_step % eval_freq == 0:
                        tr, vl = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                        train_losses.append(tr); val_losses.append(vl)

                        # MLflow 메트릭
                        try:
                            mlflow.log_metric("train_loss", tr, step=global_step)
                            mlflow.log_metric("val_loss",   vl, step=global_step)
                        except Exception as e:
                            logger.warning(f"[MLflow] metric log skipped: {e}")

                        # 상태 업데이트
                        logger.info(
                            f"Epoch {epoch+1}/{epochs} | step={global_step:06d} "
                            f"train={tr:.4f} val={vl:.4f}"
                        )
                        self.update_state(
                            state="PROGRESS",
                            meta={
                                "epoch": epoch + 1, "epochs": epochs,
                                "step": global_step,
                                "train_loss": round(tr, 4),
                                "val_loss": round(vl, 4),
                            },
                        )

                avg_epoch = epoch_loss / max(batch_count, 1)
                logger.info(f"Epoch {epoch+1} done | avg_loss={avg_epoch:.4f}")

            completed_dir = os.path.join(os.path.dirname(__file__), "..", "completed")
            os.makedirs(completed_dir, exist_ok=True)
            completed_path = os.path.join(completed_dir, f"{model_name}.pt")

            bundle = {
                "config": config,            # 요청으로 들어온 학습 설정 (dtype, context_length, tokenizer path 등)
                "layers": layer_json,        # 실제 학습에 사용한 레이어 JSON 리스트
                "state_dict": model.state_dict(),
                "vocab_size": getattr(tokenizer, "n_vocab", None),
                "meta": {
                    "run_id": run_id,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "context_length": seq_max_length,
                    "stride": stride,
                    "device": str(device),
                    "total_params": total_params,
                    "trainable_params": trainable_params,
                },
            }
            torch.save(bundle, completed_path)
            try:
                mlflow.log_artifact(completed_path, artifact_path="checkpoints")
            except Exception as e:
                logger.warning(f"[MLflow] artifact log skipped: {e}")

        logger.info("Training finished successfully.")
        return {
            "status": "success",
            "message": "Training complete",
            "train_losses": train_losses,
            "val_losses": val_losses,
            "completed_model_path": completed_path,
        }

    except Exception as e:
        logger.exception("Training failed")
        return {"status": "error", "message": str(e)}
