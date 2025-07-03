import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from model import GPTModel, GPTConfig, GPTClassifier
from classification import SpamDataset
from train_utils import save_checkpoint, load_checkpoint
import os

def train_classifier(checkpoint_path="checkpoints/model_epoch_2.pth"):
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
    dataset = SpamDataset("data/spam_classification.csv", seq_len=config.seq_len)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)  # Small batch size for CPU
    
    # Model
    device = torch.device("cpu")
    gpt_model = GPTModel(config).to(device)
    classifier = GPTClassifier(gpt_model).to(device)
    
    # Load pretrained weights
    optimizer = AdamW(classifier.parameters(), lr=1e-4)
    _, _ = load_checkpoint(gpt_model, optimizer, checkpoint_path, device)  # Load GPT weights only
    
    # Training loop
    classifier.train()
    total_correct = 0
    total_samples = 0
    losses = []
    for epoch in range(3):  # Small number for testing
        total_loss = 0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits, loss = classifier(inputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            losses.append(loss.item())
            
            # Accuracy
            preds = torch.argmax(logits, dim=-1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
        
        accuracy = total_correct / total_samples
        print(f"Epoch {epoch}, Loss: {total_loss / len(dataloader):.4f}, Accuracy: {accuracy:.4f}")
        
        # Save checkpoint
        checkpoint_path = f"checkpoints/classifier_epoch_{epoch}.pth"
        os.makedirs("checkpoints", exist_ok=True)
        save_checkpoint(classifier, optimizer, epoch, total_loss / len(dataloader), checkpoint_path)
    
    # Save loss plot
    from src.train_utils import plot_loss
    plot_loss(losses, "classifier_loss_plot.png")

if __name__ == "__main__":
    train_classifier()