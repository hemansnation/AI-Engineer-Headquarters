import pandas as pd
from unstructured.partition.text import partition_text
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import os
import yaml
import csv

def ingest_articles(config):
    input_path = config['ingestion']['input_path']
    output_path = config['ingestion']['output_path']
    
    # Read CSV with proper quoting
    df = pd.read_csv(input_path, quoting=csv.QUOTE_NONNUMERIC)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    chunks = []
    metadata = []
    for idx, row in df.iterrows():
        # Ensure content is a string
        content = str(row['content'])
        # Partition text into chunks
        elements = partition_text(text=content, max_length=512)
        for i, element in enumerate(elements):
            # Extract text from NarrativeText object
            chunk_text = element.text if hasattr(element, 'text') else str(element)
            chunks.append(chunk_text)
            metadata.append({'article_id': row['article_id'], 'chunk_id': f"{row['article_id']}_{i}"})
    
    # Encode chunks (now strings) into embeddings
    embeddings = model.encode(chunks, batch_size=32, show_progress_bar=True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    os.makedirs(output_path, exist_ok=True)
    faiss.write_index(index, os.path.join(output_path, 'faiss_index.bin'))
    with open(os.path.join(output_path, 'metadata.pkl'), 'wb') as f:
        pickle.dump(metadata, f)
    with open(os.path.join(output_path, 'chunks.pkl'), 'wb') as f:
        pickle.dump(chunks, f)

if __name__ == "__main__":
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    ingest_articles(config)