GIB = 1024 ** 3
BYTES_PER_VALUE = {'fp32': 4.0, 'fp16': 2.0, 'bf16': 2.0, 'fp8': 1.0, 'int8': 1.0, 'int4': 0.5}
def weights_gib(params_billion, precision='fp16'):
    return params_billion * 1e9 * BYTES_PER_VALUE[precision] / GIB
def kv_bytes_per_token(spec, kv_precision='fp16'):
    return 2 * spec.num_layers * spec.num_kv_heads * spec.head_dim * BYTES_PER_VALUE[kv_precision]
def kv_cache_gib(spec, context_tokens, batch_size, kv_precision='fp16'):
    return kv_bytes_per_token(spec, kv_precision) * context_tokens * batch_size / GIB
def vector_index_gib(num_vectors, dim, bytes_per_float=4):
    return num_vectors * dim * bytes_per_float / GIB
