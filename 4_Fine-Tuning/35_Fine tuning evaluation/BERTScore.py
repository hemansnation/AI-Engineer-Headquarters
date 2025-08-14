from bert_score import score

def evaluate_bertscore(model, tokenizer, data):
    model.eval()
    references = [item['response'] for item in data]
    candidates = []

    for item in tqdm(data):
        prompt = f"<s>[INST] {item['instruction']}[/INST]"
        inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=50)
        candidate = tokenizer.decode(outputs[0], skip_special_tokens=True).split('[/INST]')[1].strip()
        candidates.append(candidate)
    
    P, R, F1 = score(candidates, references, lang='en', verbose=True)
    return F1.mean().item()

if __name__ == "__main__":
    bert_f1_base = evaluate_bertscore(base_model, tokenizer, test_dataset)
    bert_f1_finetuned = evaluate_bertscore(ft_model, tokenizer, test_dataset)
    print(f"BERTScore Base: {bert_f1_base}, Fine-tuned: {bert_f1_finetuned}")