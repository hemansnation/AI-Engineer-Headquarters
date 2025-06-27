from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import io

# Load model and processor once at startup
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", use_fast=True)  # This opts into the faster image processor

model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def generate_caption(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)

    return caption
