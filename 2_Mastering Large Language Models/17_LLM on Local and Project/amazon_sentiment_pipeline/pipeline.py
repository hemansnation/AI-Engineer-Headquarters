from data_preprocessing import preprocess_data
from sentiment_classifier import run_classifier
from evaluation import evaluate_model
from visualization import plot_sentiment_distribution
import yaml

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_pipeline():

    config = load_config('configs/config.yaml')

    preprocess_data(config)
    run_classifier(config)
    evaluate_model(config)
    plot_sentiment_distribution(config)

    print("Pipeline completed successfully.")

if __name__ == "__main__":
    run_pipeline()