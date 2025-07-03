import torch
from torch.utils.data import Dataset
import pandas as pd
from tokenizer import get_tokenizer

class SpamDataset(Dataset):
    def __init__(self, csv_file, seq_len=128):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = get_tokenizer()
        self.seq_len = seq_len
        self.input_ids = []
        self.labels = []
        self._prepare_data()

    def _prepare_data(self):
        for _, row in self.data.iterrows():
            tokens = self.tokenizer(row['text'], return_tensors="pt", truncation=True, max_length=self.seq_len).input_ids[0]
            if len(tokens) < self.seq_len:
                tokens = torch.cat([tokens, torch.full((self.seq_len - len(tokens),), self.tokenizer.pad_token_id, dtype=torch.long)])
            self.input_ids.append(tokens)
            self.labels.append(row['label'])
        self.input_ids = torch.stack(self.input_ids)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.labels[idx]