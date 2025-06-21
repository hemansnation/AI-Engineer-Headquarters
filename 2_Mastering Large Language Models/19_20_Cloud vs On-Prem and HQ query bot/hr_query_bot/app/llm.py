from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import torch
from sqlmodel import Session, select
from .models import HRPolicy

class HRBot:
    def __init__(self, model_name="meta-llama/Llama-3.1-8B-Instruct"):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
        self.model.to(self.device)
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = None
        self.policy_texts = []
        self.policy_ids = []
    
    def load_policies(self, session: Session):
        policies = session.exec(select(HRPolicy)).all()
        self.policy_texts = [policy.content for policy in policies]
        self.policy_ids = [policy.id for policy in policies]
        embeddings = self.embedder.encode(self.policy_texts, convert_to_numpy=True)
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)
    
    def query(self, question: str, session: Session, top_k: int = 3):
        if not self.index:
            self.load_policies(session)
        
        query_embedding = self.embedder.encode([question], convert_to_numpy=True)
        distances, indices = self.index.search(query_embedding, top_k)
        relevant_policies = [self.policy_texts[i] for i in indices[0]]

        context = "\n".join(relevant_policies)
        prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer:"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
        )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response[len(prompt):].strip()