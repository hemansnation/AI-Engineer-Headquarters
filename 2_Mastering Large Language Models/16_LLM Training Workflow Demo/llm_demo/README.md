Production like LLM training workflow

- preprocessing code data
- fine-tuning mistral 7B (distilBERT)
- inference using flask



PS: we can build our own code generation tool

model adaptation (3 ways)
- prompt engineering
- fine-tuneing
- building it from scratch: llama, deepseek, gpt


steps:

objective
- realistic LLM training workfloww which preprocess the data, fine tune a model, deploy it for inference
- fine-tune distilBERT for a classification task (valid/invalid code snippets)
- flask API

1. folder structure
2. conda environment
3. add raw data sample to sample_code.json
4. data preprocessing
5. fine-tune distilBERT using LoRA (Low Rank Adaptation)
    - deepseek used LoRA: reduce compute cost
6. 