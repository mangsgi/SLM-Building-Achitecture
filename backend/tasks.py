from celery_worker import celery_app
import torch
import torch.nn as nn
from ml.models.factory import build_model_from_json
import os
import json
import numpy as np
import tiktoken
from celery.result import AsyncResult
import mlflow

from slm_dataset import create_dataloader_v1, load_training_data
from train_eval import calc_loss_batch, evaluate_model, generate_text

# Intel GPU 지원 추가
try:
    import intel_extension_for_pytorch as ipex
    print("Intel GPU 지원 활성화됨")
except ImportError:
    print("Intel GPU 지원을 사용할 수 없습니다. CPU로 실행됩니다.")

# 훈련 중단을 위한 전역 변수
training_should_stop = False

@celery_app.task
def train_and_infer_from_json(experiment_name, layer_json, input_text, max_length=50, temperature=0.7, top_k=40, dataset_name="tiny_shakespeare", dataset_config="default"):
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment(experiment_name)

    try:
        global training_should_stop
        training_should_stop = False
        
        # 1. 파라미터 타입 변환
        max_length = int(max_length)
        temperature = float(temperature)
        top_k = int(top_k)

        # 2. 토크나이저 초기화
        tokenizer = tiktoken.get_encoding("gpt2")
        print("토크나이저 초기화 완료")
        
        # 3. JSON 구조 검증
        if not isinstance(layer_json, list):
            raise ValueError("layer_json must be a list of layer configurations")

        # 4. 모델 초기화 및 구조 확인
        print("\n=== 모델 초기화 및 구조 확인 ===")
        try:
            model = build_model_from_json(layer_json)
            
            # 모델 구조 출력
            print("\n[모델 구조]")
            for name, module in model.named_children():
                print(f"{name}: {module}")
            
            # 모델 파라미터 수 계산
            total_params = sum(p.numel() for p in model.parameters())
            print(f"\n총 파라미터 수: {total_params:,}")
            
            # 사용자 확인 요청
            print("\n위 모델 구조가 올바른지 확인해주세요.")
            print("계속하려면 아무 키나 누르세요...")
            input()
            
        except Exception as e:
            print(f"모델 초기화 중 에러 발생: {str(e)}")
            print(f"JSON 구조: {layer_json}")
            raise e

        # 5. 데이터셋 로드
        print("\n=== 데이터셋 로드 ===")
        print(f"데이터셋 {dataset_name}/{dataset_config} 로딩 중...")
        training_text = load_training_data(dataset_name, dataset_config)
        print(f"데이터셋 로딩 완료! 텍스트 길이: {len(training_text)} 문자")

        # 6. 데이터로더 생성
        print("\n=== 데이터로더 생성 ===")
        train_loader = create_dataloader_v1(
            txt=training_text,
            batch_size=4,  # 배치 크기 감소
            max_length=64,  # 컨텍스트 길이 증가
            stride=32,     # 스트라이드 조정
            shuffle=True,
            num_workers=0
        )
        
        val_loader = create_dataloader_v1(
            txt=training_text,
            batch_size=4,
            max_length=64,
            stride=32,
            shuffle=False,
            num_workers=0
        )
        print(f"데이터로더 생성 완료! 배치 수: {len(train_loader)}")

        # 7. 디바이스 설정
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        print(f"\n=== 학습 시작 (Device: {device}) ===")

        # 옵티마이저 설정
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)

        # 학습 루프
        num_epochs = 5
        eval_freq = 2
        eval_iter = 5
        train_losses, val_losses = [], []
        tokens_seen = 0
        global_step = 0

        with mlflow.start_run():

            print("학습 시작...")
            for epoch in range(num_epochs):
                model.train()
                epoch_loss = 0
                batch_count = 0
                
                for input_batch, target_batch in train_loader:
                    optimizer.zero_grad()
                    loss = calc_loss_batch(input_batch, target_batch, model, device)
                    loss.backward()
                    optimizer.step()
                    
                    global_step += 1
                    epoch_loss += loss.item()
                    batch_count += 1

                    if global_step % eval_freq == 0:
                        train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                        train_losses.append(train_loss)
                        val_losses.append(val_loss)

                        mlflow.log_metric("train_loss", train_loss, step=global_step)
                        mlflow.log_metric("val_loss", val_loss, step=global_step)
                        
                        print(f"Epoch {epoch+1} (Step {global_step:06d}): "
                            f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

                avg_epoch_loss = epoch_loss / batch_count
                print(f"Epoch {epoch+1} 완료! 평균 Loss: {avg_epoch_loss:.3f}")

                # 중간 추론
                model.eval()
                input_ids = torch.tensor([tokenizer.encode(input_text)], device=device)
                generated_ids = generate_text(
                    model=model,
                    input_ids=input_ids,
                    max_new_tokens=max_length,
                    context_size=64,  # 컨텍스트 크기 증가
                    tokenizer=tokenizer,
                    temperature=temperature,
                    top_k=top_k
                )
                output_text = tokenizer.decode(generated_ids[0].tolist())
                print(f"[Epoch {epoch+1}] 생성된 텍스트: {output_text}")

            # 모델 저장
            ckpt_path = "trained_model.pt"
            torch.save(model.state_dict(), ckpt_path)
            mlflow.log_artifact(ckpt_path, artifact_path="checkpoints")
            print("모델 저장 완료!")

        return {
            "status": "success",
            "message": "Training and inference complete",
            "input_text": input_text,
            "generated_text": output_text,
            "train_losses": train_losses,
            "val_losses": val_losses
        }

    except Exception as e:
        print(f"오류 발생: {str(e)}")
        return {"status": "error", "message": str(e)}

@celery_app.task
def stop_training_by_id(task_id):
    """
    task_id를 사용하여 실행 중인 훈련을 중단합니다.
    """
    try:
        result = AsyncResult(task_id)
        if result.state == 'PENDING':
            return {"status": "error", "message": "Task not found or not started yet"}
        
        # Celery task 중단
        celery_app.control.revoke(task_id, terminate=True)
        return {"status": "success", "message": f"Training task {task_id} has been stopped"}
    except Exception as e:
        return {"status": "error", "message": str(e)}
