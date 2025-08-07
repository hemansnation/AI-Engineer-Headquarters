# Day 34

### What is PEFT?
Parameter Efficient Fine-tuning

- updating only a small subset of parameters rather than the entire model
- it reduces computational and memory requirements

PEFT methods
- LoRA
- Adapters
- Prefic-Tuning
- Prompt Tuning


### Low-Rank Adaptation [LoRA]

- fine-tune LLMs by adding low-rank updates to weights matrices

11, 12, 13
21, 22, 23
31, 32, 33

after row and coloumn operations

11, 12, 13
21, 22, 23
0 , 0 ,  0

- is is used in attention layer (query, value)


- [Hands-On] Fine-tune Llama with LoRA 


- QLoRA

Quantized Low Rank Adaptation

- quantizing the base model to 4 bit precision
- reducing memory usage further

- enable fine-tuning on low resource devices

quantize the base model before applying LoRA (reduce the memory by 50%)

