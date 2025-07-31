import mlflow
from celery_app import celery_app
from ml.models.factory import build_model_from_json
import torch
from .dataset import create_dataloader_v1, load_training_data
from .utils import str_to_torch_dtype

# 평가/생성 관련 함수

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
            input_ids_cond = input_ids[:, -context_size:]
            logits = model(input_ids_cond)
            next_token_logits = logits[:, -1, :]
            if top_k is not None:
                top_logits, _ = torch.topk(next_token_logits, top_k)
                min_val = top_logits[:, -1]
                next_token_logits = torch.where(
                    next_token_logits < min_val,
                    torch.tensor(float('-inf')).to(next_token_logits.device),
                    next_token_logits
                )
            if temperature > 0.0:
                next_token_logits = next_token_logits / temperature
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            if next_token.item() == tokenizer.eot_token:
                break
            input_ids = torch.cat([input_ids, next_token], dim=1)
    return input_ids

@celery_app.task(bind=True)
def train_and_infer_from_json(self, experiment_name, layer_json, input_text, max_length=50, temperature=0.7, top_k=40, dataset_name="tiny_shakespeare", dataset_config="default", dtype="fp32", epochs=5):
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment(experiment_name)
    try:
        self.update_state(state="STARTED")
        max_length = int(max_length)
        temperature = float(temperature)
        top_k = int(top_k)
        epochs = int(epochs)
        torch_dtype = str_to_torch_dtype(dtype)
        import tiktoken
        tokenizer = tiktoken.get_encoding("gpt2")
        print("토크나이저 초기화 완료")
        if not isinstance(layer_json, list):
            raise ValueError("layer_json must be a list of layer configurations")
        print("\n=== 모델 초기화 및 구조 확인 ===")
        try:
            model = build_model_from_json(layer_json, dtype=dtype)
            print("\n[모델 구조]")
            for name, module in model.named_children():
                print(f"{name}: {module}")
            total_params = sum(p.numel() for p in model.parameters())
            print(f"\n총 파라미터 수: {total_params:,}")
        except Exception as e:
            print(f"모델 초기화 중 에러 발생: {str(e)}")
            print(f"JSON 구조: {layer_json}")
            raise e
        print("\n=== 데이터셋 로드 ===")
        print(f"데이터셋 {dataset_name}/{dataset_config} 로딩 중...")
        training_text = load_training_data(dataset_name, dataset_config)
        print(f"데이터셋 로딩 완료! 텍스트 길이: {len(training_text)} 문자")
        print("\n=== 데이터로더 생성 ===")
        train_loader = create_dataloader_v1(
            txt=training_text,
            batch_size=4,
            max_length=64,
            stride=32,
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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        print(f"\n=== 학습 시작 (Device: {device}) ===")
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
        num_epochs = epochs
        eval_freq = 2
        eval_iter = 5
        train_losses, val_losses = [], []
        global_step = 0
        print("학습 시작...")
        try:
            with mlflow.start_run():
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
                    model.eval()
                    input_ids = torch.tensor([tokenizer.encode(input_text)], device=device)
                    generated_ids = generate_text(
                        model=model,
                        input_ids=input_ids,
                        max_new_tokens=max_length,
                        context_size=64,
                        tokenizer=tokenizer,
                        temperature=temperature,
                        top_k=top_k
                    )
                    output_text = tokenizer.decode(generated_ids[0].tolist())
                    print(f"[Epoch {epoch+1}] 생성된 텍스트: {output_text}")
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
        except KeyboardInterrupt:
            print("학습이 중단되었습니다!")
            return {"status": "stopped", "message": "Training was stopped by user."}
    except Exception as e:
        print(f"오류 발생: {str(e)}")
        return {"status": "error", "message": str(e)} 