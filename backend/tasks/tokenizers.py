# tasks/tokenizers.py
from __future__ import annotations
from typing import Sequence, Optional
from tokenizers import Tokenizer 
from pathlib import Path
import logging
import os
import re

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
 
# --- LLaMA-3 (HF tokenizers) ---
class Llama3TokenizerAdapter(BaseTokenizerAdapter):
    """
    Hugging Face 'tokenizers' Runtime을 사용하여 tokenizer.json을 직접 로드.
    - 대화 템플릿 없이 일반 텍스트 인/디코딩
    - bos/eos/pad id는 tokenizer.json에 정의된 스페셜 토큰을 탐색해서 설정
    """
    DEFAULT_PATH = Path(__file__).resolve().parent / "files" / "llama3-tokenizer.json"

    def __init__(self, tokenizer_file: Optional[str] = None):
        path = Path(tokenizer_file or self.DEFAULT_PATH)
        if not path.is_file():
            raise FileNotFoundError(f"LLaMA-3 tokenizer.json 파일이 필요합니다: {path}")
        self.tok = Tokenizer.from_file(str(path))

        # LLaMA3에서 흔히 쓰이는 스페셜 토큰들 탐색
        candidates = [
            "<|begin_of_text|>", "<|end_of_text|>",
            "<|start_header_id|>", "<|end_header_id|>",
            "<|eot_id|>",
        ]
        to_id = {s: self.tok.token_to_id(s) for s in candidates}
        self._bos_id = to_id.get("<|begin_of_text|>")
        self._eos_id = to_id.get("<|end_of_text|>") or to_id.get("<|eot_id|>")
        # pad는 별도 정의가 없는 경우가 대부분이므로 eos로 대체(필요 시 None으로 변경 가능)
        self._pad_id = self._eos_id

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = self.tok.encode(text).ids
        if add_bos and self._bos_id is not None:
            ids = [self._bos_id] + ids
        if add_eos and self._eos_id is not None:
            ids = ids + [self._eos_id]
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        # 학습 파이프라인에서 스페셜 토큰을 그대로 보존하려면 skip_special_tokens=False
        return self.tok.decode(list(ids), skip_special_tokens=False)

    @property
    def n_vocab(self) -> int:
        return int(self.tok.get_vocab_size())

    @property
    def bos_token_id(self): return self._bos_id
    @property
    def eos_token_id(self): return self._eos_id
    @property
    def pad_token_id(self): return self._pad_id
 
# ---- Qwen3 전용 어댑터 ----
# ---- Qwen3 전용 어댑터 (표준 방식) ----
class Qwen3TokenizerAdapter(BaseTokenizerAdapter):
    """
    Hugging Face 'tokenizers' Runtime을 사용.
    - tokenizer.json만으로 동작
    - 표준 Qwen3 인코딩 로직(스페셜 토큰 보존 + 선택적 챗 템플릿)
    """
    DEFAULT_PATH = Path(__file__).resolve().parent / "files" / "qwen3-tokenizer.json"

    # 표준 특수 토큰 목록
    _SPECIALS = [
        "<|endoftext|>",
        "<|im_start|>", "<|im_end|>",
        "<|object_ref_start|>", "<|object_ref_end|>",
        "<|box_start|>", "<|box_end|>",
        "<|quad_start|>", "<|quad_end|>",
        "<|vision_start|>", "<|vision_end|>",
        "<|vision_pad|>", "<|image_pad|>", "<|video_pad|>",
        "<think>", "</think>",
    ]
    # 스페셜 토큰을 경계로 원문을 분할해 보존 인코딩
    _SPLIT_RE = re.compile(r"(<\|[^>]+?\|>|<think>|</think>)")

    def __init__(
        self,
        tokenizer_file: Optional[str] = None,
        *,
        repo_id: Optional[str] = None,
        apply_chat_template: bool = False,
        add_generation_prompt: bool = False,
        add_thinking: bool = False,
    ):
        path = Path(tokenizer_file or self.DEFAULT_PATH)
        if not path.is_file():
            raise FileNotFoundError(
                f"Qwen3 tokenizer.json 파일이 필요합니다: {path}"
            )
        self._tok = Tokenizer.from_file(str(path))

        # 옵션 플래그 (기본은 모두 off: 일반 텍스트 태스크와 동일)
        self.apply_chat_template = apply_chat_template
        self.add_generation_prompt = add_generation_prompt
        self.add_thinking = add_thinking
        self.repo_id = repo_id

        # 스페셜 토큰 ID 맵
        self._special_to_id = {}
        for t in self._SPECIALS:
            tid = self._tok.token_to_id(t)
            if tid is not None:
                self._special_to_id[t] = tid

        # PAD/EOS 기본값: endoftext를 우선 사용
        self.pad_token_id = self._special_to_id.get("<|endoftext|>")
        self.eos_token_id = self.pad_token_id

        # 모델 변형 힌트(repo_id)로 EOS 결정 (Base는 endoftext, 그 외는 im_end)
        if repo_id and "Base" not in repo_id:
            eos_token = "<|im_end|>"
        else:
            eos_token = "<|endoftext|>"
        if eos_token in self._special_to_id:
            self.eos_token_id = self._special_to_id[eos_token]

        # BOS는 보통 정의하지 않음
        self._bos_id = None

    def _wrap_chat(self, user_msg: str) -> str:
        """
        간단한 1-turn 템플릿:
        <|im_start|>user\n{msg}<|im_end|>\n
        (+ generation 프롬프트/think 블록 선택)
        """
        s = f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        if self.add_generation_prompt:
            s += "<|im_start|>assistant"
            if self.add_thinking:
                s += "\n"  # 모델이 스스로 <think>를 낼 수 있게 빈 줄만
            else:
                # reasoning-guardrail: 명시적 thinking 블록 삽입
                s += "\n<think>\n\n</think>\n\n"
        return s

    def encode(self, text: str) -> list[int]:
        # 단일 스페셜 토큰만 들어온 경우(개행 없음) 바로 매핑
        stripped = text.strip()
        if stripped in self._special_to_id and "\n" not in stripped:
            return [self._special_to_id[stripped]]

        # 필요하면 챗 템플릿 적용
        s = self._wrap_chat(text) if self.apply_chat_template else text

        # 스페셜 토큰을 경계로 분리하여 보존 인코딩
        ids: list[int] = []
        for part in filter(None, self._SPLIT_RE.split(s)):
            if part in self._special_to_id:
                ids.append(self._special_to_id[part])
            else:
                ids.extend(self._tok.encode(part).ids)
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        # 스페셜 토큰을 그대로 노출 (학습 로그 등에서 유용)
        return self._tok.decode(list(ids), skip_special_tokens=False)

    @property
    def n_vocab(self) -> int:
        return int(self._tok.get_vocab_size())

    # 호환 속성
    @property
    def bos_token_id(self): return self._bos_id

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
        return Llama3TokenizerAdapter()  # 기본 경로: backend/tasks/files/llama3-tokenizer.json
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
    if model_id in ("llama2"):
        spm_path = str(Path(__file__).resolve().parent / "files" / "llama2-tokenizer.model")
    return choose_tokenizer(model_id, spm_path)