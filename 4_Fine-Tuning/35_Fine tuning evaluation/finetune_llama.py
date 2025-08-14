from tranformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

model_id = "meta-llama/Llama-2-7b-hf"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16"
)

base_model = AutoModelForCausalLM.from_pretrained(model_id,
                                                  quantization_config=bnb_config,
                                                  device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id)

ft_model = PeftModel.from_pretrained(base_model,
                                    "./lora_finetuned_llama")

ft_model = ft_model.merge_and_unload()