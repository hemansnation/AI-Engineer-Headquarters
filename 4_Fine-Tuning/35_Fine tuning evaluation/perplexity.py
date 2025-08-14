import torch
from tqdm import tqdm

def calculate_perplexity(model, tokenizer, data, max_length=1024):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for item in tqdm(data):
            text = item['text']
            encodings = tokenizer(text, return_tensors='pt').to(model.device)
            input_ids = encodings.input_ids[:, :max_length]
            outputs = model(input_ids, labels=input_ids)
            total_loss += outputs.loss.item()
    
    avg_loss = total_loss / len(data)
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    return perplexity

if __name__ == "__main__":
    ppl_base = calculate_perplexity(base_model, tokenizer, test_dataset)
    ppl_finetuned = calculate_perplexity(ft_model, tokenizer, test_dataset)
    print(f"Base: {ppl_base}, Fine-tuned: {ppl_finetuned}")