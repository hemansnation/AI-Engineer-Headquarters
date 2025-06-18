from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os

load_dotenv()
api_token = os.getenv("HUGGINGFACE_API_TOKEN")

if not api_token:
    raise ValueError("HUGGINGFACE_API_TOKEN not found in environment variables.")

# client = InferenceClient(token=api_token)

model_name = "meta-llama/Llama-3.1-8B-Instruct"
client = InferenceClient(model=model_name, token=api_token)


def generate_faq_answer(question):
    """
    Generate an answer to a FAQ question using a Hugging Face model."""

    system_prompt = "You are a customer support assistant for an e-commerce platform. Provide concise and helpful answers to customer questions."

    prompt = f"{system_prompt}\n\nQuestion: {question}\nAnswer:"

    try:
        response = client.text_generation(
            prompt=prompt,
            max_new_tokens=100,
            temperature=0.5,
        )
        return response.strip()
    except Exception as e:
        return f"Error generating answer: {e}"

faqs = [
    "What is the return policy for your products?",
    "How long does shipping take?",
    "Can I track my order?",
]

for question in faqs:
    answer = generate_faq_answer(question)
    print(f"Q: {question}\nA: {answer}\n")