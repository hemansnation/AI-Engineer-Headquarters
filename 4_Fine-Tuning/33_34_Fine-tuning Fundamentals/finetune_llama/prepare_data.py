import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import json
import os
import numpy as np
np.random.seed(42)

def clean_text(text):
    text = text.strip()
    text = " ".join(text.split())
    return text

def format_for_llama(row):
    prompt = f"Review: {row['text']}\nSentiment: "
    completion = "Positive" if row['label'] == 1 else "Negative"

    return {"prompt": prompt, "completion": completion}

def save_jsonl(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
    
def main():
    os.makedirs('data', exist_ok=True)

    print("Loading IMDB dataset...")
    dataset = load_dataset("imdb", split='train')

    df = dataset.to_pandas()

    print("Cleaning text...")
    df['text'] = df['text'].apply(clean_text)
    df = df.dropna()

    df = df[df['text'].str.len() > 0]

    print("Formating data for LLaMA...")
    formatted_data = df.apply(format_for_llama, axis=1).tolist()

    print("Splitting data..")

    train_data, temp_data = train_test_split(formatted_data, test_size=0.2, random_state=42)

    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

    print("Saving dataset")

    save_jsonl(train_data, 'data/train.jsonl')
    save_jsonl(val_data, 'data/validation.jsonl')
    save_jsonl(test_data, 'data/test.jsonl')

    print(f"Data Prepared! Train: {len(train_data)}, Validation: {len(val_data)}, Test: {len(test_data)}")

if __name__ == "__main__":
    main()
    