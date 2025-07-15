from transformers import pipeline
import yaml

class Generator:
    def __init__(self, config):
        self.llm = pipeline('text-generation', model=config['generation']['model'], max_length=300)
    
    def generate_answer(self, query, contexts):
        prompt = f"Question: {query}\nContext: {' '.join(contexts)}\nAnswer:"
        response = self.llm(prompt)[0]['generated_text']
        return response.split("Answer:")[1].strip()

if __name__ == "__main__":
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    generator = Generator(config)
    answer = generator.generate_answer(
        "How to reset my account password?",
        ["To reset your password, go to the login page, click 'Forgot Password,' and enter your email to receive a reset link."]
    )
    print(answer)