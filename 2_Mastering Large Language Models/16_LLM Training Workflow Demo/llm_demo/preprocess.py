import json
import os
import ast

def clean_code(code):
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False

def preprocess_data(input_path, output_path):

    with open(input_path, 'r') as f:
        data = json.load(f)
    
    cleaned_data = []
    for item in data:
        if item['label'] == 1 and clean_code(item['code']):
            cleaned_data.append(item)
        elif item['label'] == 0:
            cleaned_data.append(item)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(cleaned_data, f, indent=2)
    
    print(f"Cleaned data saved to {output_path}")

if __name__ == "__main__":
    input_path = 'data/raw/sample_code.json'
    output_path = 'data/processed/cleaned_code.json'
    preprocess_data(input_path, output_path)