# train.py
import mlflow
import torch
from model import GPTModel

def train_model(cfg: dict, exp_name):
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("slm-ui-experiment")

    model = GPTModel(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])

    with mlflow.start_run():
        mlflow.log_params({
            "vocab_size": cfg["vocab_size"],
            "emb_dim": cfg["emb_dim"],
            "context_length": cfg["context_length"],
            "n_layers": cfg["n_layers"],
            "drop_rate": cfg["drop_rate"],
            "learning_rate": cfg["learning_rate"]
        })

        for epoch in range(cfg["epochs"]):
            # 가짜 학습 데이터 (실제는 DataLoader 등 사용)
            dummy_input = torch.randint(0, cfg["vocab_size"], (4, cfg["context_length"]))
            logits = model(dummy_input)
            loss = logits.mean()  # 임시

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            mlflow.log_metric("loss", loss.item(), step=epoch)

            # Checkpoint 저장
            ckpt_path = f"model_epoch_{epoch}.pt"
            torch.save(model.state_dict(), ckpt_path)
            mlflow.log_artifact(ckpt_path, artifact_path="checkpoints")
