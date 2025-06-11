from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

def run_inference(code_snippet):
    model_path = "models/distilbert"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, torch_dtype=torch.float32)

    inputs = tokenizer(code_snippet, padding="max_length", truncation=True, max_length=128, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=1).item()
    label = "Valid" if predictions == 1 else "Invalid"
    print(f"Code snippet: {code_snippet}\nPrediction: {label}")

if __name__ == "__main__":
    test_code = "def multiply(a, b):\n    return a * b"
    run_inference(test_code)




