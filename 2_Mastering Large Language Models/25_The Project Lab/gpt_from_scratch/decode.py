import torch
import torch.nn.functional as F

def greedy_decode(model, tokenizer, prompt, max_len, device="cpu"):
    model.eval()
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    generated = input_ids
    for _ in range(max_len):
        logits, _ = model(generated)
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)
        if next_token.item() == tokenizer.eos_token_id:
            break
    return decode_tokens(tokenizer, generated[0])

def temperature_decode(model, tokenizer, prompt, max_len, temperature=0.7, device="cpu"):
    model.eval()
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    generated = input_ids
    for _ in range(max_len):
        logits, _ = model(generated)
        logits = logits[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated = torch.cat([generated, next_token], dim=1)
        if next_token.item() == tokenizer.eos_token_id:
            break
    return decode_tokens(tokenizer, generated[0])

def topk_decode(model, tokenizer, prompt, max_len, k=50, device="cpu"):
    model.eval()
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    generated = input_ids
    for _ in range(max_len):
        logits, _ = model(generated)
        probs = F.softmax(logits[:, -1, :], dim=-1)
        topk_probs, topk_indices = torch.topk(probs, k, dim=-1)
        next_token = torch.multinomial(topk_probs, num_samples=1)
        next_token = topk_indices.gather(-1, next_token)
        generated = torch.cat([generated, next_token], dim=1)
        if next_token.item() == tokenizer.eos_token_id:
            break
    return decode_tokens(tokenizer, generated[0])