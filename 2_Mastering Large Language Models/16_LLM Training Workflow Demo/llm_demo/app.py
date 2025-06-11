from flask import Flask, request, jsonify
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

app = Flask(__name__)

model_path = "models/distilbert"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path, torch_dtype=torch.float32)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    code_snippet = data.get('code', '')

    inputs = tokenizer(code_snippet, padding="max_length", truncation=True, max_length=128, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=1).item()
    label = "Valid" if predictions == 1 else "Invalid"
    return jsonify({"code": code_snippet, "prediction": label})

if __name__ == "__main__":
    app.run(port=5000)
