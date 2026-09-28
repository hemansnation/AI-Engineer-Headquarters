import os

from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

interaction = client.interactions.create(
    model=os.environ.get("GEMINI_MODEL"),
    input="Say hello to the build30 builders in a one liner"
)

print(interaction.output_text)
print(interaction.model_dump_json(indent=2, exclude_none=True))
