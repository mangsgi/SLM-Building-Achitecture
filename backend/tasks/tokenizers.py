# tasks/tokenizers.py
from __future__ import annotations
from typing import Sequence, Optional
import logging

log = logging.getLogger(__name__)

# ---- 공통 어댑터 인터페이스 ----
class BaseTokenizerAdapter:
    def encode(self, text: str) -> list[int]:
        raise NotImplementedError
    def decode(self, ids: Sequence[int]) -> str:
        raise NotImplementedError
    @property
    def n_vocab(self) -> int:
        raise NotImplementedError

# ---- Tiktoken 어댑터 ----
class TiktokenAdapter(BaseTokenizerAdapter):
    def __init__(self, enc):
        self.enc = enc
    def encode(self, text: str) -> list[int]:
        # special 토큰 이슈 회피: allowed_special="all"
        return self.enc.encode(text, allowed_special="all")
    def decode(self, ids: Sequence[int]) -> str:
        return self.enc.decode(list(ids))
    @property
    def n_vocab(self) -> int:
        return self.enc.n_vocab

# ---- SentencePiece 어댑터 (LLaMA-2) ----
class SentencePieceAdapter(BaseTokenizerAdapter):
    def __init__(self, sp):
        self.sp = sp
    def encode(self, text: str) -> list[int]:
        return self.sp.encode(text, out_type=int)
    def decode(self, ids: Sequence[int]) -> str:
        return self.sp.decode(list(ids))
    @property
    def n_vocab(self) -> int:
        return self.sp.get_piece_size()

# ---- 선택 함수 ----
def choose_tokenizer(model_name: str, spm_model_path: Optional[str] = None) -> BaseTokenizerAdapter:
    """
    model_name에 따라 토크나이저 선택:
      - gpt-2   → tiktoken 'gpt2'
      - llama-3 → tiktoken 'cl100k_base' (네가 말한 GPT-4 계열)
      - llama-2 → sentencepiece (spm 모델 경로 필요)
    """
    name = (model_name or "").lower().strip()
    if name in ("gpt", "gpt2", "gpt-2"):
        import tiktoken
        return TiktokenAdapter(tiktoken.get_encoding("gpt2"))
    elif name in ("llama3", "llama-3"):
        import tiktoken
        return TiktokenAdapter(tiktoken.get_encoding("cl100k_base"))
    elif name in ("llama2", "llama-2"):
        if not spm_model_path:
            raise ValueError("LLaMA-2 선택: tokenizer_model_path가 필요합니다.")
        try:
            import sentencepiece as spm
        except ImportError as e:
            raise RuntimeError("sentencepiece가 설치되어 있지 않습니다. pip install sentencepiece") from e
        sp = spm.SentencePieceProcessor(model_file=spm_model_path)
        return SentencePieceAdapter(sp)
    else:
        # 기본값: gpt-2
        import tiktoken
        log.warning(f"[Tokenizer] Unknown model '{model_name}', fallback to gpt2.")
        return TiktokenAdapter(tiktoken.get_encoding("gpt2"))

def choose_tokenizer_from_config(config: dict) -> BaseTokenizerAdapter:
    """config에서 model / tokenizer_model_path를 읽어 선택"""
    model_id = (config or {}).get("model", "gpt-2")
    spm_path = (config or {}).get("tokenizer_model_path")
    return choose_tokenizer(model_id, spm_path)
