import torch
import pickle
from model import GPTModel, GPTConfig
from bpe_tokenizer import BPETokenizer

# ✅ Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ✅ Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# ✅ Load model with same config used during training
config = GPTConfig(vocab_size=len(tokenizer.vocab))
model = GPTModel(config)
model.load_state_dict(torch.load("gpt_llm_trained.pth", map_location=device))
model.to(device)
model.eval()

# ✅ Inference function
def generate_text(prompt, max_new_tokens=50):
    tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)

    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits, _ = model(input_ids)
            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
            input_ids = torch.cat((input_ids, next_token), dim=1)

    output = tokenizer.decode(input_ids[0].tolist())
    return output

# ✅ CLI interface for quick testing
if __name__ == "__main__":
    prompt = input("Enter your prompt: ")
    print("📝 Generating response...")
    print(generate_text(prompt))
