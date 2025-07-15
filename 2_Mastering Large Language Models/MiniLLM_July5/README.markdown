# Building a GPT-style LLM from Scratch

## Project Overview
This project implements a GPT-style Large Language Model (LLM) from scratch using PyTorch, following a structured 3-stage approach. The goal is to create a modular, well-documented codebase suitable for learning and experimentation.

### Learning Goals
- Understand the transformer architecture and its components (multi-head attention, layer normalization, etc.).
- Implement Byte Pair Encoding (BPE) for tokenization.
- Learn to preprocess text data for next-token prediction.
- Build a training pipeline with checkpointing and fine-tuning capabilities.

## Environment Setup
For Windows 64-bit + Anaconda users:

1. Create and activate a new environment:
   ```bash
   conda create -n llm-from-scratch python=3.10 -y
   conda activate llm-from-scratch
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

4. Verify GPU support (if available):
   ```python
   import torch
   print(torch.cuda.is_available())
   ```

## Project Structure
- `stage1_llm_from_scratch.ipynb`: Implements data preprocessing and model architecture.
- `requirements.txt`: Lists required Python packages.
- Future notebooks will cover training (Stage 2) and fine-tuning/evaluation (Stage 3).

## Running in Colab
To run in Google Colab:
1. Upload the notebook and `requirements.txt`.
2. Install dependencies using `!pip install -r requirements.txt`.
3. Ensure a GPU runtime is selected for faster computation.