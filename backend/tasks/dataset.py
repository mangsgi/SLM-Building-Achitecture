# tasks/dataset.py
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
import torch

# 기본 토크나이저가 없을 때 쓸 어댑터 (tasks/tokenizers.py 필요)
try:
    from .tokenizers import TiktokenAdapter
except Exception:
    TiktokenAdapter = None  # 어댑터가 없어도 fallback 가능하도록

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        """
        tokenizer: encode(str) -> list[int], decode(list[int]) -> str, n_vocab 속성 가정
        """
        self.tokenizer = tokenizer
        self.input_ids = []
        self.target_ids = []

        # 토크나이저 종류별 인자 호환을 위해 try/except로 처리
        try:
            token_ids = tokenizer.encode(txt, allowed_special="all")
        except TypeError:
            token_ids = tokenizer.encode(txt)

        print(f"토큰화된 텍스트 길이: {len(token_ids)}")

        # 슬라이딩 윈도우로 (max_length) 시퀀스 생성
        for i in range(0, max(0, len(token_ids) - max_length), stride):
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
    txt,
    batch_size=4,
    max_length=256,
    stride=128,
    shuffle=True,
    drop_last=True,
    num_workers=0,
    tokenizer=None,
):
    """
    tokenizer를 외부에서 주입받아 사용.
    - None이면 기본으로 tiktoken gpt2를 어댑터로 감싸서 사용.
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

    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
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


def load_training_data(dataset_name, dataset_config, split="train"):
    try:
        dataset = load_dataset(dataset_name, dataset_config, split=split, trust_remote_code=True)
        return "\n".join(dataset["text"])
    except Exception as e:
        raise Exception(f"Failed to load dataset: {str(e)}")
