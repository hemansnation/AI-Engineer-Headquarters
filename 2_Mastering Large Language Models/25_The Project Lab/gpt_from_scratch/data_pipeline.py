import torch
from datasets import load_dataset
from transformers import GPT2Tokenizer
import os

class FineWebDataset:
    def __init__(self, data_dir, split="train", seq_len=128, max_samples=1000):
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.tokenizer = self._init_tokenizer()
        self.data = self._load_data(split, max_samples)
        self.input_ids, self.target_ids = self._prepare_data()

    def _init_tokenizer(self):
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token  # Use <|endoftext|> as padding
        return tokenizer

    def _load_data(self, split, max_samples):
        dataset = load_dataset("HuggingFaceFW/fineweb", "sample-10BT", split=split, streaming=True)
        data = []
        for i, sample in enumerate(dataset):
            if i >= max_samples:
                break
            data.append(sample["text"])
        return data

    def _prepare_data(self):
        input_ids = []
        target_ids = []
        for text in self.data:
            tokens = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=self.seq_len).input_ids[0]
            if len(tokens) < self.seq_len:
                tokens = torch.cat([tokens, torch.full((self.seq_len - len(tokens),), self.tokenizer.pad_token_id, dtype=torch.long)])
            for i in range(0, len(tokens) - self.seq_len, self.seq_len):
                input_seq = tokens[i:i + self.seq_len]
                target_seq = tokens[i + 1:i + self.seq_len + 1]
                input_ids.append(input_seq)
                target_ids.append(target_seq)
        return torch.stack(input_ids), torch.stack(target_ids)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]