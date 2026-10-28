import pytest
from sizing.presets import PRESETS, spec_from_config
from sizing.memory import weights_gib, kv_bytes_per_token, kv_cache_gib, vector_index_gib
from sizing.traffic import Traffic, queries_per_day, avg_qps, peak_qps, concurrent_requests, tokens_per_day
from sizing.planner import Gpu, make_plan
LLAMA = PRESETS['llama-3.1-8b']
def test_weights_fp16_is_two_bytes_per_parameter():
    assert weights_gib(8.03, 'fp16') == pytest.approx(14.96, abs=0.01)
def test_int4_is_a_quarter_of_fp16():
    assert weights_gib(8.03, 'int4') == pytest.approx(weights_gib(8.03, 'fp16') / 4)
def test_kv_bytes_per_token_for_llama_8b():
    assert kv_bytes_per_token(LLAMA) == 131072
def test_kv_cache_for_one_8k_request_is_one_gib():
    assert kv_cache_gib(LLAMA, 8192, 1) == pytest.approx(1.0)
def test_fp8_kv_cache_is_half():
    assert kv_cache_gib(LLAMA, 8192, 1, 'fp8') == pytest.approx(0.5)
def test_vector_index_for_one_million_384_dim_vectors():
    assert vector_index_gib(1_000_000, 384) == pytest.approx(1.43, abs=0.01)
def test_traffic_numbers():
    t = Traffic(10000, 8, 1500, 300)
    assert queries_per_day(t) == 80000
    assert avg_qps(t) == pytest.approx(2.222, abs=0.001)
    assert peak_qps(t) == pytest.approx(6.667, abs=0.001)
    assert concurrent_requests(t) == pytest.approx(53.33, abs=0.01)
    assert tokens_per_day(t)['total'] == 144_000_000
def test_plan_picks_the_slowest_resource():
    t = Traffic(10000, 8, 1500, 300)
    gpu = Gpu('test', 80, 2.5, 2500, 20000)
    plan = make_plan(LLAMA, t, gpu)
    assert plan.gpus_for_memory == 1
    assert plan.gpus_for_decode == 2
    assert plan.bottleneck == 'decode speed'
    assert plan.gpus_final == 3
    assert plan.monthly_cost == pytest.approx(3 * 2.5 * 730)
def test_spec_from_config_reads_nested_text_config():
    cfg = {'text_config': {'num_attention_heads': 32, 'num_key_value_heads': 8, 'hidden_size': 4096, 'num_hidden_layers': 32}}
    spec = spec_from_config('demo', cfg, 8.03e9)
    assert (spec.num_layers, spec.num_kv_heads, spec.head_dim) == (32, 8, 128)
def test_spec_from_config_defaults_kv_heads_to_attention_heads():
    cfg = {'num_attention_heads': 16, 'hidden_size': 2048, 'num_hidden_layers': 24}
    assert spec_from_config('demo', cfg, 1e9).num_kv_heads == 16
