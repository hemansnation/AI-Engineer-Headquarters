import torch
from model import GPTModel, GPTConfig
from tokenizer import get_tokenizer
from decode import greedy_decode, temperature_decode, topk_decode
from train_utils import load_checkpoint

def evaluate(checkpoint_path, prompt="The quick brown fox"):
    # Config
    config = GPTConfig(
        vocab_size=50257,
        seq_len=128,
        embed_dim=128,
        num_heads=4,
        num_layers=2,
        dropout=0.1
    )
    
    # Model and Tokenizer
    device = torch.device("cpu")
    model = GPTModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters())  # Dummy optimizer for checkpoint loading
    tokenizer = get_tokenizer()
    
    # Load checkpoint
    epoch, loss = load_checkpoint(model, optimizer, checkpoint_path, device)
    
    # Generate samples
    model.eval()
    print(f"Evaluating checkpoint from epoch {epoch} (Loss: {loss:.4f})")
    print("Greedy Decoding:", greedy_decode(model, tokenizer, prompt, max_len=50, device=device))
    print("Temperature Decoding (T=0.7):", temperature_decode(model, tokenizer, prompt, max_len=50, device=device))
    print("Top-k Decoding (k=50):", topk_decode(model, tokenizer, prompt, max_len=50, k=50, device=device))

if __name__ == "__main__":
    evaluate("checkpoints/model_epoch_2.pth")