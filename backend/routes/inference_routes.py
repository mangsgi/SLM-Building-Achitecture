from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from pathlib import Path
import torch
import json
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
        # 1) 모델 파일(.pt 또는 .pth) 로드
        model_name = req.model_name
        pt_path = Path("completed") / f"{model_name}.pt"
        pth_path = Path("completed") / f"{model_name}.pth"

        model_path = None
        if pt_path.exists():
            model_path = pt_path
        elif pth_path.exists():
            model_path = pth_path
        
        if model_path is None:
            raise HTTPException(status_code=404, detail=f"모델 파일(.pt 또는 .pth)을 찾을 수 없습니다: {model_name}")

        loaded_data = torch.load(model_path, map_location="cpu")

        # 시나리오 분기:
        # 1) 번들(dict)인 경우: layers, config, state_dict 추출
        # 2) state_dict만 있는 경우: json에서 layers, config 로드
        if isinstance(loaded_data, dict) and "state_dict" in loaded_data and "layers" in loaded_data:
            # 번들 시나리오
            layers = loaded_data.get("layers")
            config = loaded_data.get("config", {}) or {}
            state_dict = loaded_data.get("state_dict")
        else:
            # state_dict 단독 시나리오
            state_dict = loaded_data
            
            # 해당 모델의 구조(.json) 파일 찾기
            structure_path = Path("temp_structures") / f"{model_name}.json"
            if not structure_path.exists():
                raise HTTPException(
                    status_code=404, 
                    detail=f"모델 구조 파일({structure_path.name})을 찾을 수 없습니다. "
                           "이 모델은 state_dict만 포함하고 있어 구조 파일이 반드시 필요합니다."
                )
            
            with open(structure_path, "r", encoding="utf-8") as f:
                structure = json.load(f)
            
            # 구조 파일에서 config와 layers 분리
            config = structure[0] if isinstance(structure, list) and len(structure) > 0 else {}
            layers = structure[1:] if isinstance(structure, list) and len(structure) > 1 else []

        if not isinstance(layers, list) or not layers or state_dict is None:
            raise HTTPException(status_code=400, detail="모델 구조(layers) 또는 가중치(state_dict)를 로드할 수 없습니다.")

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
            # (안전) 최대 생성 길이 재계산: 프롬프트 길이를 뺀 나머지
            prompt_len = input_ids.size(1)
            gen_length = min(max_len_req, max(0, context_length - prompt_len))

            # 캐시 사용 가능 여부 확인: 모델에 forward_cached가 있으면 사용
            use_cached_path = hasattr(model, "forward_cached")
            
            caches = None           # 레이어별 KV 캐시를 dict나 tuple로 보관(모델 구현에 따름)
            start_pos = 0           # RoPE 오프셋(프리필 시작은 0)
            logits = None            

            # ---- 7-1) 프리필: 프롬프트 전체를 한 번에 통과시키며 캐시 채우기 ----
            if use_cached_path:
                # 모델 쪽 forward_cached가 어텐션 레이어들에 start_pos/kv_cache를 전파해 준다는 전제
                logits, caches = model.forward_cached(
                    input_ids,            # (B, T_prompt)
                    caches=None,          # 처음엔 캐시 없음
                    start_pos=start_pos,  # 0부터 시작
                    use_cache=True,       # 캐시 채우기/유지
                    return_logits=True    # (필요하면) 마지막 출력 반환
                )
                start_pos = prompt_len   # 다음 스텝의 시작 위치는 프롬프트 길이
            else:
                # 기존 방식(비효율)으로도 동작 가능하게 폴백
                logits = model(input_ids[:, -context_length:])
                                    
            # ---- 7-2) 증분: 토큰을 1개씩 만들며 KV 캐시 재사용 ----
            for _ in range(gen_length):
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
                    next_token = torch.multinomial(probs, 1)    # (B, 1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                # 시퀀스에 새 토큰 붙이기
                input_ids = torch.cat([input_ids, next_token], dim=1)

                if use_cached_path:
                    # 증분 스텝: 새 토큰 1개만 모델에 넣고, 캐시와 start_pos를 넘김
                    logits, caches = model.forward_cached(
                        next_token,         # (B, 1)
                        caches=caches,      # 이전 캐시 재사용
                        start_pos=start_pos,# 지금까지 쌓인 전체 길이(프리필 길이 + 생성 토큰 수)
                        use_cache=True,
                        return_logits=True
                    )
                    start_pos += 1         # 다음 스텝은 위치 1 증가
                else:
                    # 폴백: 기존처럼 마지막 context_length만 잘라서 전체 재계산(느림)
                    logits = model(input_ids[:, -context_length:])

                # (옵션) 컨텍스트 초과 방지: 초과하면 중단
                if input_ids.size(1) >= context_length:
                    break

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
