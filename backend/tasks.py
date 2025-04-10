# backend/tasks.py

from celery_worker import celery_app  # ✅ Celery 인스턴스를 직접 가져옴 (순환 참조 없음)
from ml.config import GPT_CONFIG_124M  # 기본 config 값 불러오기
from model.model_factor import GPTModel  # GPT 모델 클래스 가져오기
import torch

@celery_app.task(name="train_gpt_model")  # ✅ Celery 인스턴스로 직접 태스크 등록
def train_gpt_model(layer_config: dict, config: dict = GPT_CONFIG_124M):
    """
    모델 객체를 구성하고, 해당 모델의 구조를 출력하여 반환하는 작업.
    """
    try:
        # 모델 구성만 진행 (학습은 하지 않음)
        model = GPTModel(config, layer_config)

        # Dummy input 생성 (context_length에 맞춰서)
        dummy_input = torch.randint(0, config["vocab_size"], (1, config["context_length"]))

        # forward 패스를 실행하여 모델이 잘 동작하는지 확인
        output = model(dummy_input)

        # 모델 구조 출력만 반환
        model_structure = str(model)
        return f"✅ Model built successfully. Structure:\n\n{model_structure}\n\nOutput Shape: {output.shape}"

    except Exception as e:
        return f"Model building failed: {str(e)}"
