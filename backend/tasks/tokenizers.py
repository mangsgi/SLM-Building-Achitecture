# tasks/tokenizers.py
from __future__ import annotations
from typing import Sequence, Optional
from tokenizers import Tokenizer 
from pathlib import Path
import logging
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
 
# ---- Qwen3 전용 어댑터 (로컬 tokenizer.json 사용) ----
class Qwen3TokenizerAdapter(BaseTokenizerAdapter):
    """
    Hugging Face 'tokenizers' Runtime을 직접 사용.
    - config 없이 로컬 tokenizer.json만 고정 경로로 로드
    - 챗 템플릿 등은 전혀 안 씀 (GPT-2 등과 동일한 일반 텍스트 테스크)
    """
    # 고정 경로: backend/tasks/files/qwen3-tokenizer.json 에 두세요.
    DEFAULT_PATH = Path(__file__).resolve().parent / "files" / "qwen3-tokenizer.json"

    def __init__(self, tokenizer_file: Optional[str] = None):
        path = Path(tokenizer_file or self.DEFAULT_PATH)
        if not path.is_file():
            raise FileNotFoundError(
                f"Qwen3 tokenizer.json 파일이 필요합니다: {path}\n"
                f"(원한다면 경로를 바꾸려면 Qwen3TokenizerAdapter(DEFAULT_PATH=...) 수정)"
            )
        self.tok = Tokenizer.from_file(str(path))

        # (선택) HF 어댑터와 인터페이스 맞춤: bos/eos id 노출
        # Qwen3 계열은 보통 다음 특수 토큰을 가짐
        specials = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
        self._special_to_id = {s: self.tok.token_to_id(s) for s in specials}
        # eos 우선순위: im_end 있으면 그걸, 없으면 endoftext
        self.eos_token_id = self._special_to_id.get("<|im_end|>") \
                            if self._special_to_id.get("<|im_end|>") is not None \
                            else self._special_to_id.get("<|endoftext|>")
        # pad는 없는 경우가 많으니 eos로 대체
        self.pad_token_id = self._special_to_id.get("<|endoftext|>")

    def encode(self, text: str) -> list[int]:
        # 특수토큰은 tokenizer.json 안에 정의되어 있으면 그대로 id가 나옵니다.
        return self.tok.encode(text).ids

    def decode(self, ids: Sequence[int]) -> str:
        return self.tok.decode(list(ids), skip_special_tokens=False)

    @property
    def n_vocab(self) -> int:
        return int(self.tok.get_vocab_size())

    # HF 어댑터와 동일한 속성명 제공(데이터셋 쪽에서 getattr로 쓰는 경우 대비)
    @property
    def bos_token_id(self):  # Qwen3 기본 bos는 없을 수 있음
        return None
    @property
    def eos_token_id_(self):  # 내부용으로 갖고 있지만 호환 위해 별칭 안 씀
        return self.eos_token_id

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
        # tiktoken gpt2
        import tiktoken
        return TiktokenAdapter(tiktoken.get_encoding("gpt2"))
    elif name in ("llama3"):
        # tiktoken cl100k_base
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
    elif name in ("qwen3"):
        # Qwen3 토크나이저 어댑터 사용
        return Qwen3TokenizerAdapter()  # 기본 경로: backend/files/qwen3-tokenizer.json
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