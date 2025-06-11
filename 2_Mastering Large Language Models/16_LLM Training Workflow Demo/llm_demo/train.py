from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import torch
import numpy as np

def fine_tune_model():
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2, torch_dtype=torch.float32)

    dataset = load_dataset("json", data_files="data/processed/cleaned_code.json", split="train")

    def tokenize_function(examples):
        return tokenizer(examples['code'], padding="max_length", truncation=True, max_length=128)
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
    tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=['q_lin', 'v_lin'],
        lora_dropout=0.1,
        bias="none",
        task_type="SEQ_CLS"
    )
    model = get_peft_model(model, lora_config)

    training_args = TrainingArguments(
        output_dir="models/distilbert",
        evaluation_strategy="no",
        per_device_train_batch_size=1,
        num_train_epochs=1,
        logging_steps=10,
        save_steps=100,
        remove_unused_columns=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )

    print("Starting training...")
    print(f"Model: {model_name}")
    print(f"Dataset size: {len(tokenized_dataset)}")
    trainer.train()

    model.save_pretrained("models/distilbert")
    tokenizer.save_pretrained("models/distilbert")
    print("Model saved to models/distilbert")

if __name__ == "__main__":
    fine_tune_model()
    print("Fine-tuning complete.")
