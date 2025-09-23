# tasks/dataset.py
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
import torch
from typing import List, Optional, Sequence, Union

# 기본 토크나이저가 없을 때 쓸 어댑터 (tasks/tokenizers.py 필요)
try:
    from .tokenizers import TiktokenAdapter  # tiktoken용
except Exception:
    TiktokenAdapter = None  # 어댑터가 없어도 fallback 가능하도록

def _safe_encode(tokenizer, text: str) -> List[int]:
    """tiktoken/SentencePiece 겸용 인코딩 (allowed_special 인자 차이 흡수)"""
    try:
        return tokenizer.encode(text, allowed_special="all")
    except TypeError:
        return tokenizer.encode(text)

def _maybe_id(obj, name: str) -> Optional[int]:
    """어댑터가 bos_token_id/eos_token_id를 제공하면 쓰고, 없으면 None"""
    try:
        return getattr(obj, name, None)
    except Exception:
        return None


class DatasetV1(Dataset):
    def __init__(
        self,
        txt: Union[str, Sequence[str]],
        tokenizer,
        max_length: int,
        stride: int,
        *,
        add_bos: bool = False,
        add_eos_between_docs: bool = True,
        bos_id: Optional[int] = None,
        eos_id: Optional[int] = None,
    ):
        """
        tokenizer: encode(str) -> list[int], decode(list[int]) -> str, n_vocab 속성 가정
        txt: 단일 문자열 또는 문서 리스트
        add_bos: 각 문서 앞에 BOS 토큰 삽입
        add_eos_between_docs: 문서 경계마다 EOS 삽입
        bos_id/eos_id: 명시하지 않으면 토크나이저 어댑터가 제공(가능한 경우)
        """
        self.tokenizer = tokenizer
        self.input_ids = []
        self.target_ids = []

        # 토크나이저가 노출하는 BOS/EOS ID가 있으면 기본값으로 사용
        if bos_id is None:
            bos_id = _maybe_id(tokenizer, "bos_token_id")
        if eos_id is None:
            eos_id = _maybe_id(tokenizer, "eos_token_id")

        # txt를 문서 리스트로 정규화
        if isinstance(txt, str):
            docs: List[str] = [txt]
        else:
            docs = list(txt)

        # 문서별 토큰화 → (옵션) BOS/EOS 삽입 → 하나로 이어붙이기
        flat_ids: List[int] = []
        total_chars = 0
        for doc in docs:
            total_chars += len(doc)
            ids = _safe_encode(tokenizer, doc)
            if add_bos and bos_id is not None:
                flat_ids.append(bos_id)
            flat_ids.extend(ids)
            if add_eos_between_docs and eos_id is not None:
                flat_ids.append(eos_id)

        token_ids = flat_ids
        print(f"토큰화된 텍스트 길이: {len(token_ids)} (문자 길이 합: {total_chars})")

        # 슬라이딩 윈도우로 (max_length) 시퀀스 생성
        # causal LM 표준: 입력 x[0:T] → 타깃 y[1:T+1]
        N = len(token_ids)
        for i in range(0, max(0, N - max_length), stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk, dtype=torch.long))
            self.target_ids.append(torch.tensor(target_chunk, dtype=torch.long))

        print(f"생성된 데이터셋 크기: {len(self.input_ids)} 샘플")

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(
    txt: Union[str, Sequence[str]],
    batch_size=4,
    max_length=256,
    stride=128,
    shuffle=True,
    drop_last=True,
    num_workers=0,
    tokenizer=None,
    *,
    add_bos: bool = False,
    add_eos_between_docs: bool = True,
    bos_id: Optional[int] = None,
    eos_id: Optional[int] = None,
):
    """
    tokenizer를 외부에서 주입받아 사용.
    - None이면 기본으로 tiktoken gpt2를 어댑터로 감싸서 사용.
    - txt는 문자열 또는 문서 리스트를 허용(범용 Causal LM 전처리)
    """
    if tokenizer is None:
        # 기본 토크나이저: tiktoken gpt2
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        if TiktokenAdapter is not None:
            tokenizer = TiktokenAdapter(enc)
        else:
            # 어댑터가 없으면 원시 enc를 그대로 사용 (encode만 호출해 씀)
            tokenizer = enc

    vocab_size_log = getattr(tokenizer, "n_vocab", None)
    print(f"토크나이저 초기화 완료. 어휘 크기: {vocab_size_log if vocab_size_log is not None else 'N/A'}")

    dataset = DatasetV1(
        txt,
        tokenizer,
        max_length,
        stride,
        add_bos=add_bos,
        add_eos_between_docs=add_eos_between_docs,
        bos_id=bos_id,
        eos_id=eos_id,
    )
    print(f"데이터셋 생성 완료. 샘플 수: {len(dataset)}")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )
    print(f"데이터로더 생성 완료. 배치 수: {len(dataloader)}")
    return dataloader


def load_training_data(
    dataset_name,
    dataset_config="default",
    split="train",
    dataset_text_field=None,
    streaming=False,
    max_rows=None,
    *,
    return_list: bool = False,  # True면 문서 리스트 반환, False면 하나의 문자열
):
    """
    dataset_text_field: 원본 텍스트 컬럼명(없으면 'text' 시도)
    streaming: True면 스트리밍 모드로 순회하며 수집
    max_rows: None이 아니면 해당 수 만큼만 모아 반환(디버깅/샘플링용)
    return_list: True이면 List[str], False이면 '\n'.join(...) 문자열 반환
    """
    dataset = load_dataset(
        dataset_name, dataset_config, split=split,
        streaming=streaming, trust_remote_code=True
    )

    # 어떤 데이터든 최종적으로 'text'를 확보
    def _collect_rows(ds_iter):
        texts = []
        for i, row in enumerate(ds_iter):
            txt = row.get(dataset_text_field or "text", "")
            if txt:
                texts.append(txt)
            if max_rows and len(texts) >= max_rows:
                break
        return texts

    if streaming:
        texts = _collect_rows(dataset)
    else:
        if dataset_text_field and dataset_text_field != "text":
            if dataset_text_field in dataset.column_names:
                dataset = dataset.rename_column(dataset_text_field, "text")
            else:
                raise ValueError(f"'{dataset_text_field}' column not found")
        texts = list(dataset["text"]) if max_rows is None else list(dataset["text"][:max_rows])

    # 반환 형식 선택(뒤로 호환 위해 기본은 문자열)
    if return_list:
        return texts
    return "\n".join(texts)
