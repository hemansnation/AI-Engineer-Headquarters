import torch
import torch.nn as nn
import pickle
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from bpe_tokenizer import BPETokenizer
from model import GPTModel, GPTConfig

# ✅ Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# ✅ Custom Dataset
class TextDataset(Dataset):
    def __init__(self, text, tokenizer, seq_length):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.tokens = tokenizer.encode(text)

    def __len__(self):
        return len(self.tokens) - self.seq_length

    def __getitem__(self, idx):
        input_ids = self.tokens[idx:idx + self.seq_length]
        target_ids = self.tokens[idx + 1:idx + self.seq_length + 1]
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(target_ids, dtype=torch.long)

# ✅ Corpus and Tokenizer
corpus = ['hello world', 'how are you', 'fine thanks', 'world peace']
tokenizer = BPETokenizer(vocab_size=300)
tokenizer.train(corpus)

# ✅ Training Data
sample_text = ' '.join(corpus * 100)  # Expand corpus
seq_length = 10
dataset = TextDataset(sample_text, tokenizer, seq_length)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# ✅ Model Setup
config = GPTConfig(vocab_size=len(tokenizer.vocab), seq_len=seq_length)
model = GPTModel(config).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

# ✅ Training Loop
model.train()
for epoch in range(5):
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        logits, loss = model(inputs, targets)
        
        if torch.isnan(loss):
            print("🚨 NaN loss detected — stopping early")
            break

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}: Loss = {loss.item():.4f}")

# ✅ Save model and tokenizer
torch.save(model.state_dict(), "gpt_llm_trained.pth")
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

print("✅ Model and tokenizer saved successfully.")
