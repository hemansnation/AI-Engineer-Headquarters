import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from model import GPTModel, GPTConfig
from instruction import QADataset
from train_utils import save_checkpoint, load_checkpoint
from decode import greedy_decode
from tokenizer import get_tokenizer
import os

def train_instruction(checkpoint_path="checkpoints/model_epoch_2.pth"):
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
    dataset = QADataset("data/qa_instructions.json", seq_len=config.seq_len)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)  # Small batch size
    
    # Model
    device = torch.device("cpu")
    model = GPTModel(config).to(device)
    
    # Freeze lower layers (token embedding and first transformer block)
    for param in model.token_emb.parameters():
        param.requires_grad = False
    for param in model.blocks[0].parameters():
        param.requires_grad = False
    
    # Load pretrained weights
    optimizer = AdamW(model.parameters(), lr=1e-4)
    _, _ = load_checkpoint(model, optimizer, checkpoint_path, device)
    
    # Training loop
    model.train()
    tokenizer = get_tokenizer()
    losses = []
    for epoch in range(3):
        total_loss = 0
        for inputs, targets, masks in dataloader:
            inputs, targets, masks = inputs.to(device), targets.to(device), masks.to(device)
            optimizer.zero_grad()
            logits, loss = model(inputs, targets)
            # Apply attention mask to loss
            loss = loss * masks.view(-1)
            loss = loss.sum() / masks.sum()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            losses.append(loss.item())
        
        print(f"Epoch {epoch}, Loss: {total_loss / len(dataloader):.4f}")
        
        # Generate sample output
        prompt = "Instruction: What is the capital of France? Response:"
        print(f"Sample Output: {greedy_decode(model, tokenizer, prompt, max_len=20, device=device)}")
        
        # Save checkpoint
        checkpoint_path = f"checkpoints/instruction_epoch_{epoch}.pth"
        os.makedirs("checkpoints", exist_ok=True)
        save_checkpoint(model, optimizer, epoch, total_loss / len(dataloader), checkpoint_path)
    
    # Save loss plot
    from src.train_utils import plot_loss
    plot_loss(losses, "instruction_loss_plot.png")

if __name__ == "__main__":
    train_instruction()