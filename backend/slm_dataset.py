import torch
from torch.utils.data import DataLoader, Dataset
import tiktoken
from datasets import load_dataset


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        print(f"토큰화된 텍스트 길이: {len(token_ids)}")

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk, dtype=torch.long))
            self.target_ids.append(torch.tensor(target_chunk, dtype=torch.long))

        print(f"생성된 데이터셋 크기: {len(self.input_ids)} 샘플")

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True, num_workers=0):
    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")
    print(f"토크나이저 초기화 완료. 어휘 크기: {tokenizer.n_vocab}")

    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    print(f"데이터셋 생성 완료. 샘플 수: {len(dataset)}")

    # Create dataloader
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        drop_last=drop_last, 
        num_workers=num_workers
    )
    print(f"데이터로더 생성 완료. 배치 수: {len(dataloader)}")

    return dataloader


def load_training_data(dataset_name, dataset_config, split="train"):
    """
    Hugging Face datasets에서 데이터를 로드하는 함수
    """
    try:
        dataset = load_dataset(dataset_name, dataset_config, split=split, trust_remote_code=True)
        return "\n".join(dataset["text"])
    except Exception as e:
        raise Exception(f"Failed to load dataset: {str(e)}")