import torch
from torch.utils.data import Dataset
import json
from src.tokenizer import get_tokenizer

class QADataset(Dataset):
    def __init__(self, json_file, seq_len=128):
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        self.tokenizer = get_tokenizer()
        self.seq_len = seq_len
        self.input_ids = []
        self.target_ids = []
        self.attention_masks = []
        self._prepare_data()

    def _prepare_data(self):
        for item in self.data:
            instruction = item['instruction']
            response = item['response']
            text = f"Instruction: {instruction} Response: {response} <|endoftext|>"
            tokens = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=self.seq_len)
            input_ids = tokens.input_ids[0]
            attention_mask = tokens.attention_mask[0]
            
            if len(input_ids) < self.seq_len:
                pad_len = self.seq_len - len(input_ids)
                input_ids = torch.cat([input_ids, torch.full((pad_len,), self.tokenizer.pad_token_id, dtype=torch.long)])
                attention_mask = torch.cat([attention_mask, torch.zeros(pad_len, dtype=torch.long)])
            
            target_ids = input_ids.clone()
            self.input_ids.append(input_ids)
            self.target_ids.append(target_ids)
            self.attention_masks.append(attention_mask)
        
        self.input_ids = torch.stack(self.input_ids)
        self.target_ids = torch.stack(self.target_ids)
        self.attention_masks = torch.stack(self.attention_masks)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx], self.attention_masks[idx]