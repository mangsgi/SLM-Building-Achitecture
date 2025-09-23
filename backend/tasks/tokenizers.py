# tasks/tokenizers.py
from __future__ import annotations
from typing import Sequence, Optional
import logging
from pathlib import Path
import os

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
    if name in ("gpt-2"):
        import tiktoken
        return TiktokenAdapter(tiktoken.get_encoding("gpt2"))
    elif name in ("llama3"):
        import tiktoken
        return TiktokenAdapter(tiktoken.get_encoding("cl100k_base"))
    elif name in ("llama2"):
        # SentencePiece 로드
        if spm_model_path is None:
            raise ValueError(
                "llama-2 토크나이저를 사용하려면 spm_model_path (예: '.../tokenizer.model')를 지정하세요."
            )
        if not os.path.isfile(spm_model_path):
            raise FileNotFoundError(f"SentencePiece 모델 파일을 찾을 수 없습니다: {spm_model_path}")

        import sentencepiece as spm
        sp = spm.SentencePieceProcessor()
        ok = sp.load(spm_model_path)
        if not ok:
            raise RuntimeError(f"SentencePiece 모델 로드 실패: {spm_model_path}")
        return SentencePieceAdapter(sp)
    else:
        raise ValueError(f"Unknown model '{model_name}''s tokenizer when choosing tokenizer.")

def choose_tokenizer_from_config(config: dict) -> BaseTokenizerAdapter:
    """config에서 model / tokenizer_model_path를 읽어 선택"""
    model_id = (config or {}).get("model")
    if model_id is None:
        raise ValueError("model is required in config when choosing tokenizer.")
    spm_path = None
    if model_id not in ("llama2"):
        spm_path = str(Path(__file__).resolve().parent / "files" / "llama2-tokenizer.model")
    return choose_tokenizer(model_id, spm_path)