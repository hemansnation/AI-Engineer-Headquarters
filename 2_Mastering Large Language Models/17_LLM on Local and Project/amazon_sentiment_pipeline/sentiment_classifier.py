import pandas as pd
import ollama
import yaml
import logging
from tqdm import tqdm

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def classify_sentiment(text, model_name, max_tokens):
    prompt = f"Classify the sentiment of the following text as positive, neutral, or negative: {text}\nSentiment:"

    try:
        response = ollama.generate(
            model=model_name,
            prompt=prompt,
            options={
                "max_tokens": max_tokens
            }
        )
        sentiment = response['response'].strip().lower()

        if sentiment in ['positive', 'neutral', 'negative']:
            return sentiment
        return 'neutral'
    except Exception as e:
        logging.error(f"Error classifying sentiment: {e}")
        return 'neutral'

def run_classifier(config):
    logging.basicConfig(filename=config['output']['logs_path'], level=logging.INFO)
    logging.info("Starting sentiment classification...")

    df = pd.read_csv(config['data']['processed_path'])
    model_name = config['model']['ollama_model']
    max_tokens = config['model']['max_tokens']

    tqdm.pandas()
    df['predicted_sentiment'] = df['cleaned_text'].progress_apply(
        lambda x: classify_sentiment(x, model_name, max_tokens)
    )

    Path()(config['output']['results_path']).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(config['output']['results_path'], index=False)
    logging.info("Sentiment classification completed successfully.")
    return df

if __name__ == "__main__":
    config = load_config('configs/config.yaml')
    run_classifier(config)