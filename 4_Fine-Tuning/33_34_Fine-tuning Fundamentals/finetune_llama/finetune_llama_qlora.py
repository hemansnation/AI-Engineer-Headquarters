import os
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import torch
from trasformers import BitsAndBytesConfig

import numpy as np
np.random.seed(42)
torch.manual_seed(42)

def main():
    model_name = "meta-llama/Llama-2-7b-hf"
    data_path = 'data/'
    output_dir = 'output/finetuned_llama'

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    print("Loading Llama model and tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quantization_config, device_map="auto")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("Loading dataset...")
    dataset = load_dataset('json', data_files={
        'train': os.path.join(data_path, 'train.jsonl'),
        'validation': os.path.join(data_path, 'validation.jsonl')
    })

    def tokenize_function(examples):
        inputs = tokenizer(examples['prompt'], truncation=True, padding='max_length', max_length=512)
        inputs['labels'] = tokenizer(examples['completion'], truncation=True, padding='max_length', max_length=16)['input_ids']
        return inputs

    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=['prompt', 'completion'])

    print("Configuring QLoRA...")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        evaluation_strategy="steps",
        eval_steps=500,
        save_steps=500,
        learning_rate=2e-5,
        logging_steps=100,
        max_steps=1000,
        save_total_limit=2,
        report_to="none",
    )

    print('Starting QLoRA fine-tuning...')

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['validation']
    )

    trainer.train()
    print("Fine-tuning completed!")
    print("Saving the model...")

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("Sample prediction: ")
    test_prompt = "Review: This movie was fantastic!\nSentiment: "
    inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=10)
    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(prediction)

if __name__ == "__main__":
    main()


