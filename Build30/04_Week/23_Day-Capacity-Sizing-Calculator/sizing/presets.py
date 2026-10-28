import json
from dataclasses import dataclass
@dataclass(frozen=True)
class ModelSpec:
    name: str
    params_billion: float
    num_layers: int
    num_attention_heads: int
    num_kv_heads: int
    head_dim: int
PRESETS = {
    'llama-3.1-8b': ModelSpec('llama-3.1-8b', 8.03, 32, 32, 8, 128),
    'llama-3.1-70b': ModelSpec('llama-3.1-70b', 70.6, 80, 64, 8, 128),
    'mistral-7b-v0.3': ModelSpec('mistral-7b-v0.3', 7.25, 32, 32, 8, 128),
    'qwen2.5-7b': ModelSpec('qwen2.5-7b', 7.62, 28, 28, 4, 128),
}
def spec_from_config(name, config, total_params):
    cfg = config.get('text_config', config)
    heads = cfg['num_attention_heads']
    kv_heads = cfg.get('num_key_value_heads', heads)
    head_dim = cfg.get('head_dim') or cfg['hidden_size'] // heads
    return ModelSpec(name, total_params / 1e9, cfg['num_hidden_layers'], heads, kv_heads, head_dim)
def load_from_hub(repo_id):
    from huggingface_hub import HfApi, hf_hub_download
    path = hf_hub_download(repo_id, 'config.json')
    with open(path, encoding='utf-8') as file:
        config = json.load(file)
    info = HfApi().model_info(repo_id, expand=['safetensors'])
    if info.safetensors is None:
        raise ValueError('No parameter count found for ' + repo_id + ', add it by hand as a ModelSpec')
    return spec_from_config(repo_id, config, info.safetensors.total)
