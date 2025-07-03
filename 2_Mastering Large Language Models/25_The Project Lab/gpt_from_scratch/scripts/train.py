import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from ..data_pipeline import FineWebDataset
from ..model import GPTModel, GPTConfig
from ..tokenizer import get_tokenizer
from ..train_utils import save_checkpoint, load_checkpoint, plot_loss
from ..decode import greedy_decode, temperature_decode, topk_decode
import os

def train(resume_from_checkpoint=None):
    # Config
    config = GPTConfig(
        vocab_size=50257,
        seq_len=128,
        embed_dim=128,
        num_heads=4,
        num_layers=2,
        dropout=0.1
    )
    
    # Data
    dataset = FineWebDataset(data_dir="data", seq_len=config.seq_len, max_samples=1000)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # Model and Optimizer
    device = torch.device("cpu")
    model = GPTModel(config).to(device)
    optimizer = AdamW(model.parameters(), lr=3e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=len(dataloader) * 3)  # 3 epochs
    tokenizer = get_tokenizer()
    
    # Resume from checkpoint if provided
    start_epoch = 0
    losses = []
    if resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
        start_epoch, _ = load_checkpoint(model, optimizer, resume_from_checkpoint, device)
    
    # Training loop
    model.train()
    for epoch in range(start_epoch, 3):
        total_loss = 0
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            _, loss = model(inputs, targets)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            losses.append(loss.item())
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        # Save checkpoint
        checkpoint_path = f"checkpoints/model_epoch_{epoch}.pth"
        os.makedirs("checkpoints", exist_ok=True)
        save_checkpoint(model, optimizer, epoch, total_loss / len(dataloader), checkpoint_path)
        
        # Evaluate by sampling
        prompt = "The quick brown fox"
        print(f"\nEpoch {epoch} Sample Generations:")
        print("Greedy:", greedy_decode(model, tokenizer, prompt, max_len=20, device=device))
        print("Temperature:", temperature_decode(model, tokenizer, prompt, max_len=20, device=device))
        print("Top-k:", topk_decode(model, tokenizer, prompt, max_len=20, k=50, device=device))
        
        print(f"Epoch {epoch} Average Loss: {total_loss / len(dataloader):.4f}")
    
    # Save loss plot
    plot_loss(losses, "loss_plot.png")

if __name__ == "__main__":
    train()