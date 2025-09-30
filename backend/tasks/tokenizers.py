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
 
# --- LLaMA-3 (tiktoken .model) ---
class Llama3TokenizerAdapter(BaseTokenizerAdapter):
    """
    tiktoken 기반의 LLaMA-3 토크나이저.
    - Meta Llama-3 tokenizer.model(.tiktoken 포맷) 직접 로드
    - special 토큰 ID는 Meta tokenizer.json 기준 하드코딩
    - encode(text) 는 BOS/EOS를 자동으로 붙이지 않음(참고 구현과 동일)
      * 필요시 encode_with_flags(text, bos=True, eos=True) 제공
    """
    DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "files" / "llama3-tokenizer.model"

    def __init__(self, model_path: Optional[str] = None):
        import tiktoken
        from tiktoken.load import load_tiktoken_bpe

        path = Path(model_path or self.DEFAULT_MODEL_PATH)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{path}")

        mergeable = load_tiktoken_bpe(str(path))

        # Meta Llama-3 tokenizer.json 기준
        self.special = {
            "<|begin_of_text|>": 128000,
            "<|end_of_text|>": 128001,
            "<|start_header_id|>": 128006,
            "<|end_header_id|>": 128007,
            "<|eot_id|>": 128009,
        }
        # reserved_* 채우기
        self.special.update({
            f"<|reserved_{i}|>": 128002 + i
            for i in range(256)
            if 128002 + i not in self.special.values()
        })

        pat_str = (
            r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
            r"|[^\r\n\p{L}\p{N}]?\p{L}+"
            r"|\p{N}{1,3}"
            r"| ?[^\s\p{L}\p{N}]+[\r\n]*"
            r"|\s*[\r\n]+"
            r"|\s+(?!\S)"
            r"|\s+"
        )

        self.model = tiktoken.Encoding(
            name=path.name,
            pat_str=pat_str,
            mergeable_ranks=mergeable,
            special_tokens=self.special,
        )

        # 편의용 ID들
        self._bos_id = self.special["<|begin_of_text|>"]
        self._eos_id = self.special["<|end_of_text|>"]
        self._pad_id = self._eos_id  # pad 미정의 → eos 대체

    # 참고 구현과 동일: BOS/EOS 자동 미부착
    def encode(self, text: str) -> list[int]:
        # 참고 구현은 allowed_special 지정 없이 encode → 스페셜이 문자열에 있으면 에러 가능
        # 완전 동일 동작을 원하면 아래 한 줄 사용:
        return self.model.encode(text)

        # 만약 스페셜 문자열을 허용하고 싶다면 위를 아래로 바꾸면 됨:
        # return self.model.encode(text, allowed_special="all")

    # 참고 구현의 보조 메서드와 동일 동작 (옵션)
    def encode_with_flags(self, text: str, bos: bool = False, eos: bool = False) -> list[int]:
        ids = ([self._bos_id] if bos else []) + self.model.encode(text)
        if eos:
            ids.append(self._eos_id)
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        return self.model.decode(list(ids))

    @property
    def n_vocab(self) -> int:
        return self.model.n_vocab

    @property
    def bos_token_id(self): return self._bos_id
    @property
    def eos_token_id(self): return self._eos_id
    @property
    def pad_token_id(self): return self._pad_id

 
# ---- Qwen3 전용 어댑터 ----
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