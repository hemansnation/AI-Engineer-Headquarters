import torch
from model import GPTModel, GPTConfig
from tokenizer import get_tokenizer
from decode import greedy_decode
from train_utils import load_checkpoint

def evaluate_instruction(checkpoint_path="checkpoints/instruction_epoch_2.pth"):
    # Config
    config = GPTConfig(
        vocab_size=50257,
        seq_len=128,
        embed_dim=128,
        num_heads=4,
        num_layers=2,
        dropout=0.1
    )
    
    # Model
    device = torch.device("cpu")
    model = GPTModel(config).to(device)
    optimizer = AdamW(model.parameters())  # Dummy optimizer
    tokenizer = get_tokenizer()
    
    # Load checkpoint
    epoch, loss = load_checkpoint(model, optimizer, checkpoint_path, device)
    
    # Test prompts
    model.eval()
    prompts = [
        "Instruction: What is the capital of France? Response:",
        "Instruction: Who wrote 'Pride and Prejudice'? Response:",
        "Instruction: What is 2 + 2? Response:"
    ]
    for prompt in prompts:
        output = greedy_decode(model, tokenizer, prompt, max_len=20, device=device)
        print(f"Prompt: {prompt}\nOutput: {output}\n")

if __name__ == "__main__":
    evaluate_instruction()