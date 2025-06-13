import pandas as pd
import nltk
import spacy
import yaml
import logging
from pathlib import Path

nltk.download('stopwords')
nlp = spacy.load("en_core_web_sm")

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def preprocess_text(text, max_length):
    if not isinstance(text, str):
        return ""
    
    doc = nlp(text.lower()[:max_length], disable=["ner", "lemmatizer"])
    tokens = [token.text for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)

def preprocess_data(config):
    logging.basicConfig( filename=config['output']['logs_path'],level=logging.INFO)
    logging.info("Starting data preprocessing...")

    chuck_size = 100000

    chucks = pd.read_csv(
        config['data']['raw_path'],
        sep='\t',
        usecols=[config['data']['columns']['review_text'], config['data']['columns']['rating']],
        chunksize=chuck_size,
        low_memory=False,
    )

    df = next(chucks).dropna().sample(n=config['preprocessing']['sample_size'], random_state=42)

    df['sentiment'] = df[config['data']['columns']['rating']].apply(
        lambda x: 'positive' if x >= 4 else 'negative' if x <= 2 else 'neutral'
    )

    df['cleaned_text'] = df[config['data']['columns']['review_text']].apply(
        lambda x: preprocess_text(x, config['preprocessing']['max_length'])
    )

    Path(config['data']['processed_path']).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(config['data']['processed_path'], index=False)

    logging.info("Data preprocessing completed successfully. Processed {len(df) reviews}")
    return df

if __name__ == "__main__":
    config = load_config('configs/config.yaml')
    preprocess_data(config)
