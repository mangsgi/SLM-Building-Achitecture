from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from pathlib import Path
import torch
from ml.models.factory import build_model_from_json
from tasks.tokenizers import choose_tokenizer_from_config

router = APIRouter()

class InferenceRequest(BaseModel):
    model_name: str
    input_text: str
    max_length: int = 50
    temperature: float = 0.7
    top_k: int = 40


@router.post("/generate-text", tags=["Inference"])
def generate_text_api(req: InferenceRequest):
    try:
        # 1) 번들(.pt) 로드
        model_path = Path("completed") / f"{req.model_name}.pt"
        if not model_path.exists():
            raise HTTPException(status_code=404, detail="모델 번들(.pt)을 찾을 수 없습니다.")

        bundle = torch.load(model_path, map_location="cpu")
        if not isinstance(bundle, dict):
            raise HTTPException(status_code=400, detail="번들 포맷이 올바르지 않습니다(dict 아님).")

        layers = bundle.get("layers")
        config = bundle.get("config", {}) or {}
        state_dict = bundle.get("state_dict")

        if not isinstance(layers, list) or state_dict is None:
            raise HTTPException(status_code=400, detail="번들에 layers/state_dict가 없습니다.")

        dtype = config.get("dtype", "fp32")
        context_length = int(config.get("context_length", 128))

        # 2) 모델 복원
        model = build_model_from_json(layers, dtype=dtype)
        model.load_state_dict(state_dict)
        model.eval()

        # 3) 토크나이저 복원
        try:
            tokenizer = choose_tokenizer_from_config(config)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"토크나이저 초기화 실패: {e}")

        # 4) 입력 인코딩
        try:
            input_ids_list = tokenizer.encode(req.input_text)
        except TypeError:
            input_ids_list = tokenizer.encode(req.input_text, allowed_special="all")

        # 텐서로 변환
        input_ids = torch.tensor([input_ids_list], dtype=torch.long)

        # (안전장치 A) 프롬프트가 context_length를 넘으면 오른쪽 기준으로 슬라이스
        if input_ids.size(1) > context_length:
            input_ids = input_ids[:, -context_length:]

        # 5) 디바이스
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        input_ids = input_ids.to(device)

        # 6) 하이퍼파라미터 정리 + (안전장치 B) 생성 길이 캡
        if req.max_length <= 0:
            raise HTTPException(status_code=400, detail="max_length는 1 이상이어야 합니다.")
        max_len_req = int(req.max_length)
        gen_length = min(max_len_req, context_length)  # 컨텍스트 길이 이상 생성하지 않도록 캡

        top_k = int(req.top_k) if req.top_k is not None else 0
        if top_k < 0:
            top_k = 0
        temperature = float(req.temperature) if req.temperature is not None else 1.0
        if temperature < 0:
            temperature = 0.0  # 음수 방지

        # 7) 생성 루프
        with torch.no_grad():
            for _ in range(gen_length):
                # 항상 최근 context_length 토큰만 모델에 투입 (메모리/성능 안정)
                logits = model(input_ids[:, -context_length:])
                next_token_logits = logits[:, -1, :]

                # top-k 필터링
                if top_k > 0:
                    top_vals, _ = torch.topk(next_token_logits, top_k)
                    min_val = top_vals[:, -1].unsqueeze(-1)
                    next_token_logits = torch.where(
                        next_token_logits < min_val,
                        torch.full_like(next_token_logits, float("-inf")),
                        next_token_logits,
                    )

                # temperature/샘플링
                if temperature > 0:
                    probs = torch.softmax(next_token_logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, 1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                input_ids = torch.cat([input_ids, next_token], dim=1)

        # 8) 디코딩 (CPU로 이동 후 디코딩 권장)
        output_ids = input_ids[0].detach().cpu().tolist()
        output_text = tokenizer.decode(output_ids)

        return {
            "status": "success",
            "input_text": req.input_text,
            "output_text": output_text
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"추론 중 오류: {e}")
