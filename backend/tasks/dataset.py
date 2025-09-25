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
        max_total_tokens: int | None = None
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride

        if bos_id is None:
            bos_id = _maybe_id(tokenizer, "bos_token_id")
        if eos_id is None:
            eos_id = _maybe_id(tokenizer, "eos_token_id")

        if isinstance(txt, str):
            docs: List[str] = [txt]
        else:
            docs = list(txt)

        flat_ids: List[int] = []
        total_chars = 0
        for doc in docs:
            total_chars += len(doc)
            ids = _safe_encode(tokenizer, doc)
            if add_bos and bos_id is not None:
                flat_ids.append(bos_id)
            # flat_ids.extend(ids)
            # 토큰을 누적하되 상한을 넘지 않도록
            for tid in ids:
                flat_ids.append(tid)
                if max_total_tokens is not None and len(flat_ids) >= max_total_tokens + 1:
                    break
            if add_eos_between_docs and eos_id is not None:
                flat_ids.append(eos_id)
            if max_total_tokens is not None and len(flat_ids) >= max_total_tokens + 1:
                break
        
        # 전체 토큰을 하나의 긴 텐서로 저장 (메모리 효율적)
        self.token_ids = torch.tensor(flat_ids, dtype=torch.long)
        print(f"토큰화된 텍스트 길이: {len(self.token_ids)} (문자 길이 합: {total_chars})")

        # 데이터셋의 총 샘플 수 미리 계산
        self.num_samples = (len(self.token_ids) - self.max_length) // self.stride
        print(f"생성된 데이터셋 크기: {self.num_samples} 샘플")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 요청된 idx에 해당하는 데이터 조각을 동적으로 생성
        start_idx = idx * self.stride
        input_chunk = self.token_ids[start_idx : start_idx + self.max_length]
        target_chunk = self.token_ids[start_idx + 1 : start_idx + self.max_length + 1]
        return input_chunk, target_chunk


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
    max_total_tokens: int | None = None,
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
        max_total_tokens=max_total_tokens,
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
