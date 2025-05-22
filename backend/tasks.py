from celery_worker import celery_app
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from ml.models.factory import build_model_from_json
import os
import json
import numpy as np
import tiktoken
from datasets import load_dataset
from celery.result import AsyncResult

# Intel GPU 지원 추가
try:
    import intel_extension_for_pytorch as ipex
    print("Intel GPU 지원 활성화됨")
except ImportError:
    print("Intel GPU 지원을 사용할 수 없습니다. CPU로 실행됩니다.")

# 훈련 중단을 위한 전역 변수
training_should_stop = False


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        print(f"토큰화된 텍스트 길이: {len(token_ids)}")

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk, dtype=torch.long))
            self.target_ids.append(torch.tensor(target_chunk, dtype=torch.long))

        print(f"생성된 데이터셋 크기: {len(self.input_ids)} 샘플")

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True, num_workers=0):
    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")
    print(f"토크나이저 초기화 완료. 어휘 크기: {tokenizer.n_vocab}")

    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    print(f"데이터셋 생성 완료. 샘플 수: {len(dataset)}")

    # Create dataloader
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        drop_last=drop_last, 
        num_workers=num_workers
    )
    print(f"데이터로더 생성 완료. 배치 수: {len(dataloader)}")

    return dataloader

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss

def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
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

def generate_text(model, input_ids, max_new_tokens, context_size, tokenizer, temperature=0.0, top_k=None):
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # 컨텍스트 크기만큼만 사용
            input_ids_cond = input_ids[:, -context_size:]
            logits = model(input_ids_cond)
            next_token_logits = logits[:, -1, :]

            # top-k 샘플링
            if top_k is not None:
                top_logits, _ = torch.topk(next_token_logits, top_k)
                min_val = top_logits[:, -1]
                next_token_logits = torch.where(
                    next_token_logits < min_val,
                    torch.tensor(float('-inf')).to(next_token_logits.device),
                    next_token_logits
                )

            # temperature 스케일링
            if temperature > 0.0:
                next_token_logits = next_token_logits / temperature
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            # EOS 토큰 체크
            if next_token.item() == tokenizer.eot_token:
                break

            input_ids = torch.cat([input_ids, next_token], dim=1)

    return input_ids

def load_training_data(dataset_name, dataset_config, split="train"):
    """
    Hugging Face datasets에서 데이터를 로드하는 함수
    """
    try:
        dataset = load_dataset(dataset_name, dataset_config, split=split, trust_remote_code=True)
        return "\n".join(dataset["text"])
    except Exception as e:
        raise Exception(f"Failed to load dataset: {str(e)}")

@celery_app.task
def train_and_infer_from_json(layer_json, input_text, max_length=50, temperature=0.7, top_k=40, dataset_name="tiny_shakespeare", dataset_config="default"):
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
        torch.save(model.state_dict(), "trained_model.pt")
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
