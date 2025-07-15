from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from datasets import Dataset
import yaml
import os
import torch
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline  # Updated import

def evaluate_rag(query, contexts, answer, reference, config):
    # Initialize transformers pipeline with max_new_tokens
    transformers_pipeline = pipeline(
        'text-generation',
        model=config['generation']['model'],
        max_length=500,  # Use max_new_tokens instead of max_length
        device="mps" if torch.backends.mps.is_available() else "cpu"
    )
    
    # Wrap pipeline in langchain_huggingface's HuggingFacePipeline
    llm = HuggingFacePipeline(pipeline=transformers_pipeline)
    
    # Load embeddings model from cache if available
    cache_dir = os.path.expanduser('~/.cache/huggingface/sentence_transformers')
    model_path = os.path.join(cache_dir, 'all-MiniLM-L6-v2')
    embeddings_model = model_path if os.path.exists(model_path) else 'all-MiniLM-L6-v2'
    
    dataset = Dataset.from_dict({
        'question': [query],
        'contexts': [contexts],
        'answer': [answer],
        'reference': [reference]
    })
    
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=llm,
        embeddings=embeddings_model
    )
    return result

if __name__ == "__main__":
    # Suppress tokenizers parallelism warning
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    result = evaluate_rag(
        query="How to reset my account password?",
        contexts=["To reset your password, go to the login page, click 'Forgot Password,' and enter your email to receive a reset link."],
        answer="Click 'Forgot Password' on the login page and enter your email.",
        reference="To reset your password, go to the login page, click 'Forgot Password,' enter your email, and follow the reset link.",
        config=config
    )
    print(result)