import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from pathlib import Path

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def plot_sentiment_distribution(config):

    df = pd.read_csv(config['output']['results_path'])

    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x='predicted_sentiment', order=['positive', 'neutral', 'negative'], palette='viridis')
    plt.title('Sentiment Distribution')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')

    Path(config['output']['figures_path']).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(f"{config['output']['figures_path']}/sentiment_distribution.png")
    plt.close()


if __name__ == "__main__":
    config = load_config('configs/config.yaml')
    plot_sentiment_distribution(config)
    