import torch
from transformers import GPT2LMHeadModel
from tokenizer import get_tokenizer
from decode import greedy_decode, temperature_decode, topk_decode

def compare_gpt2(prompt="The quick brown fox"):
    device = torch.device("cpu")
    tokenizer = get_tokenizer()
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    model.eval()
    
    print("Pretrained GPT-2 Generations:")
    print("Greedy:", greedy_decode(model, tokenizer, prompt, max_len=50, device=device))
    print("Temperature (T=0.7):", temperature_decode(model, tokenizer, prompt, max_len=50, device=device))
    print("Top-k (k=50):", topk_decode(model, tokenizer, prompt, max_len=50, k=50, device=device))

if __name__ == "__main__":
    compare_gpt2()