# requirements:
# pip install transformers tiktoken

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from pathlib import Path
import torch, json
from ml.models.factory import build_model_from_json          # ← (.pt/.pth)용 기존 경로는 유지
from tasks.tokenizers import choose_tokenizer_from_config    # ← 토크나이저는 기존 방식 사용
from transformers import GPT2Config, GPT2LMHeadModel

router = APIRouter()

class InferenceRequest(BaseModel):
    model_name: str
    input_text: str
    max_length: int = 50
    temperature: float = 0.7
    top_k: int = 40


# ---------------------------
# 공통 유틸
# ---------------------------

def _resolve_model_file(model_name: str) -> tuple[Path, str]:
    """
    completed/{name}{.pt|.pth|.bin} 파일을 찾아 반환.
    return: (경로, 파일명 stem)
    """
    base = Path("completed")
    name = model_name.strip()

    # 사용자가 확장자를 포함한 경우
    direct = base / name
    if direct.exists() and direct.suffix in {".pt", ".pth", ".bin"}:
        return direct, direct.stem

    # 확장자 없이 들어온 경우
    for ext in (".pt", ".pth", ".bin"):
        p = base / f"{name}{ext}"
        if p.exists():
            return p, p.stem

    raise HTTPException(404, detail=f"모델 파일(.pt/.pth/.bin)을 찾을 수 없습니다: {model_name}")


# ---------------------------
# === BIN PATH (HF GPT-2) ===
# ---------------------------

def _load_hf_gpt2_cfg_from_json(structure_json: Path) -> GPT2Config:
    """
    temp_structures/{stem}.json 에서 HF 스타일의 GPT-2 config(dict)를 읽어 GPT2Config 생성.
    (딕셔너리가 없거나 키가 부족하면 GPT-2 small 기본값으로 폴백)
    """
    if not structure_json.exists():
        # 같은 이름의 구조 JSON이 반드시 있어야 한다는 요구대로 에러 처리
        raise HTTPException(404, detail=f"구조 JSON을 찾을 수 없습니다: {structure_json}")

    try:
        raw = json.loads(structure_json.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(400, detail=f"구조 JSON 파싱 실패: {e}")

    if not isinstance(raw, dict):
        # 예전 내부 포맷([config, ...layers])이 들어왔다면, 첫 원소가 dict일 가능성 고려
        if isinstance(raw, list) and raw and isinstance(raw[0], dict):
            raw = raw[0]
        else:
            raise HTTPException(400, detail="구조 JSON이 HF config(dict) 형식이 아닙니다.")

    # HF gpt2 config 키들을 우선 사용, 없으면 안전한 기본값
    return GPT2Config(
        n_layer=raw.get("num_hidden_layers", raw.get("n_layer", 12)),
        n_head=raw.get("num_attention_heads", raw.get("n_head", 12)),
        n_embd=raw.get("n_embd", raw.get("hidden_size", 768)),
        n_positions=raw.get("n_positions", raw.get("max_position_embeddings", 1024)),
        vocab_size=raw.get("vocab_size", 50257),
        layer_norm_epsilon=raw.get("layer_norm_eps", 1e-5),
        bos_token_id=raw.get("bos_token_id", 50256),
        eos_token_id=raw.get("eos_token_id", 50256),
        # 필요한 필드 더 있으면 여기에 추가
    )


def _build_model_and_tokenizer_from_bin(bin_path: Path, stem: str):
    """
    .bin(state_dict) + temp_structures/{stem}.json(HF gpt2 config dict) → HF GPT2 모델 구성
    토크나이저는 'model=gpt-2'로 choose_tokenizer_from_config를 호출해 로딩.
    """
    # 1) config
    cfg_path = Path("temp_structures") / f"{stem}.json"
    gpt2_cfg = _load_hf_gpt2_cfg_from_json(cfg_path)

    # 2) model
    state_dict = torch.load(bin_path, map_location="cpu")
    if not isinstance(state_dict, dict):
        raise HTTPException(400, detail=f"{bin_path.name}에서 state_dict(dict)를 읽지 못했습니다.")
    model = GPT2LMHeadModel(gpt2_cfg)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    # 필요하면 로그로 확인만 (실행은 계속)
    if unexpected:
        print("[BIN] unexpected keys:", unexpected)
    if missing:
        print("[BIN] missing keys:", missing)
    model.eval()

    # 3) tokenizer: 요청대로 새 config에 model='gpt-2' 넣어 사용
    try:
        tokenizer = choose_tokenizer_from_config({"model": "gpt-2"})
    except Exception as e:
        raise HTTPException(400, detail=f"토크나이저 초기화 실패(gpt-2): {e}")

    # 4) context length
    context_length = int(getattr(gpt2_cfg, "n_positions", 1024))

    return model, tokenizer, context_length


def _generate_with_hf_model(model, tokenizer, text: str, *, max_length: int, temperature: float, top_k: int, context_length: int):
    # 인코딩
    try:
        ids = tokenizer.encode(text)
    except TypeError:
        ids = tokenizer.encode(text)

    input_ids = torch.tensor([ids], dtype=torch.long)
    if input_ids.size(1) > context_length:
        input_ids = input_ids[:, -context_length:]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    input_ids = input_ids.to(device)

    if max_length <= 0:
        raise HTTPException(400, detail="max_length는 1 이상이어야 합니다.")
    temperature = max(0.0, float(temperature if temperature is not None else 1.0))
    top_k = max(0, int(top_k if top_k is not None else 0))

    prompt_len = input_ids.size(1)
    max_new = min(int(max_length), max(0, context_length - prompt_len)) or 1

    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=max_new,
            temperature=temperature,
            do_sample=(temperature > 0.0),
            top_k=top_k,
            eos_token_id=getattr(model.config, "eos_token_id", 50256),
            pad_token_id=getattr(model.config, "pad_token_id", 0),
        )

    return tokenizer.decode(out[0].detach().cpu().tolist())


# ---------------------------
# 기존(.pt/.pth) 경로 유틸
# ---------------------------

def _load_layers_config_state_bundle_or_statedict(model_path: Path, base_name: str):
    """
    (.pt/.pth) 전용 로더: 기존 로직 유지
    """
    loaded = torch.load(model_path, map_location="cpu")

    if isinstance(loaded, dict) and "state_dict" in loaded and "layers" in loaded:
        layers = loaded.get("layers")
        config = loaded.get("config", {}) or {}
        state_dict = loaded.get("state_dict")
    else:
        state_dict = loaded
        structure_path = Path("temp_structures") / f"{base_name}.json"
        if not structure_path.exists():
            raise HTTPException(
                404,
                detail=f"모델 구조 파일({structure_path.name})을 찾을 수 없습니다. "
                       f"이 모델은 state_dict만 포함하고 있어 구조 파일이 반드시 필요합니다."
            )
        structure = json.loads(structure_path.read_text(encoding="utf-8"))
        config = structure[0] if isinstance(structure, list) and len(structure) > 0 else {}
        layers = structure[1:] if isinstance(structure, list) and len(structure) > 1 else []

    if not isinstance(layers, list) or not layers or state_dict is None:
        raise HTTPException(400, detail="모델 구조(layers) 또는 가중치(state_dict)를 로드할 수 없습니다.")
    return layers, config, state_dict


def _generate_with_internal_model(layers, config, state_dict, *, text, max_length, temperature, top_k):
    dtype = config.get("dtype", "fp32")
    context_length = int(config.get("context_length", 128))

    model = build_model_from_json(layers, dtype=dtype)
    model.load_state_dict(state_dict)
    model.eval()

    try:
        tokenizer = choose_tokenizer_from_config(config)
    except Exception as e:
        raise HTTPException(400, detail=f"토크나이저 초기화 실패: {e}")

    # 아래는 기존 생성 루프 그대로 (요약)
    try:
        ids = tokenizer.encode(text)
    except TypeError:
        ids = tokenizer.encode(text, allowed_special="all")
    input_ids = torch.tensor([ids], dtype=torch.long)
    if input_ids.size(1) > context_length:
        input_ids = input_ids[:, -context_length:]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device); input_ids = input_ids.to(device)
    if max_length <= 0:
        raise HTTPException(400, detail="max_length는 1 이상이어야 합니다.")
    max_len_req = int(max_length)
    gen_length = min(max_len_req, max(0, context_length - input_ids.size(1)))
    top_k = max(0, int(top_k or 0))
    temperature = max(0.0, float(temperature if temperature is not None else 1.0))

    with torch.no_grad():
        use_cached_path = hasattr(model, "forward_cached")
        start_pos = 0; logits = None
        if use_cached_path:
            logits, caches = model.forward_cached(input_ids, None, 0, True, True)
            start_pos = input_ids.size(1)
        else:
            logits = model(input_ids[:, -context_length:])
        for _ in range(gen_length):
            nxt = logits[:, -1, :]
            if top_k > 0:
                top_vals, _ = torch.topk(nxt, top_k)
                min_val = top_vals[:, -1].unsqueeze(-1)
                nxt = torch.where(nxt < min_val, torch.full_like(nxt, float("-inf")), nxt)
            if temperature > 0:
                probs = torch.softmax(nxt / temperature, dim=-1)
                tok = torch.multinomial(probs, 1)
            else:
                tok = torch.argmax(nxt, dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, tok], dim=1)
            if use_cached_path:
                logits, caches = model.forward_cached(tok, caches, start_pos, True, True); start_pos += 1
            else:
                logits = model(input_ids[:, -context_length:])
            if input_ids.size(1) >= context_length: break

    return choose_tokenizer_from_config(config).decode(input_ids[0].cpu().tolist())


# ---------------------------
# 라우터
# ---------------------------

@router.post("/generate-text", tags=["Inference"])
def generate_text_api(req: InferenceRequest):
    try:
        model_path, stem = _resolve_model_file(req.model_name)

        # BIN이면 HF GPT-2 경로로
        if model_path.suffix == ".bin":
            model, tokenizer, context_len = _build_model_and_tokenizer_from_bin(model_path, stem)
            output = _generate_with_hf_model(
                model, tokenizer, req.input_text,
                max_length=req.max_length,
                temperature=req.temperature,
                top_k=req.top_k,
                context_length=context_len,
            )
            return {"status": "success", "input_text": req.input_text, "output_text": output}

        # 그 외(.pt/.pth)는 기존 경로
        layers, config, state_dict = _load_layers_config_state_bundle_or_statedict(model_path, stem)
        output = _generate_with_internal_model(
            layers, config, state_dict,
            text=req.input_text, max_length=req.max_length,
            temperature=req.temperature, top_k=req.top_k,
        )
        return {"status": "success", "input_text": req.input_text, "output_text": output}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, detail=f"추론 중 오류: {e}")
