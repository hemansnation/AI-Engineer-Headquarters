import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from model import GPTModel  # import your model class
from tokenizer import BPETokenizer  # import your tokenizer
import pickle

# Load preprocessed data (assumes it's saved in a .pkl file or torch tensor)
with open("train_data.pkl", "rb") as f:
    train_data = pickle.load(f)

# Hyperparameters
block_size = 128
batch_size = 64
epochs = 10
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Create dataset & dataloader
class LLMTrainDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx:idx + self.block_size], dtype=torch.long)
        y = torch.tensor(self.data[idx + 1:idx + 1 + self.block_size], dtype=torch.long)
        return x, y

train_dataset = LLMTrainDataset(train_data, block_size)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Initialize model
model = GPTModel(vocab_size=tokenizer.vocab_size, block_size=block_size).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# Training loop
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        x, y = [b.to(device) for b in batch]
        logits = model(x)
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

# Save the model
torch.save(model.state_dict(), "gpt_llm_trained.pth")
