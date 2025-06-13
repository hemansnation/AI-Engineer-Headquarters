import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report
import yaml
import logging

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def evaluate_model(config):
    logging.basicConfig(filename=config['output']['logs_path'], level=logging.INFO)
    logging.info("Starting model evaluation...")

    df = pd.read_csv(config['output']['results_path'])

    y_true = df['sentiment']
    y_pred = df['predicted_sentiment']

    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    report = classification_report(y_true, y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Classification Report:\n", report)

    logging.info(f"Accuracy: {accuracy:.4f}")
    logging.info(f"F1 Score: {f1:.4f}")
    logging.info("Classification Report:\n" + report)
    logging.info("Model evaluation completed successfully.")

if __name__ == "__main__":
    config = load_config('configs/config.yaml')
    evaluate_model(config)