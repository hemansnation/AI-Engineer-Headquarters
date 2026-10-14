import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from extractor import extract
from llm import TICKS, FakeProvider
from schemas import TicketExtraction, to_api_schema

GOOD = {
    'customer_name': 'Rahul Verma',
    'order_id': 'ORD-20417',
    'category': 'billing',
    'urgency': 'medium',
    'sentiment': 'negative',
    'amount_usd': 49.99,
    'summary': 'Customer was charged twice for one order.',
    'requested_action': 'Refund one payment',
}

def test_clean_reply_passes_on_first_attempt():
    provider = FakeProvider([json.dumps(GOOD)])
    result = extract('some email', provider)
    assert result.ok
    assert result.attempts == 1
    assert result.data.order_id == 'ORD-20417'

def test_bad_order_id_triggers_one_retry_with_error_in_prompt():
    bad = dict(GOOD, order_id='20417')
    provider = FakeProvider([json.dumps(bad), json.dumps(GOOD)])
    result = extract('some email', provider)
    assert result.ok
    assert result.attempts == 2
    assert 'order_id' in provider.prompts[1]
    assert 'ORD-48213' in provider.prompts[1]

def test_never_valid_stops_at_the_cap_without_crashing():
    provider = FakeProvider(['not json at all'])
    result = extract('some email', provider, max_attempts=3)
    assert not result.ok
    assert result.attempts == 3
    assert provider.calls == 3
    assert result.data is None

def test_code_fence_around_json_is_accepted():
    fenced = TICKS + 'json\n' + json.dumps(GOOD) + '\n' + TICKS
    result = extract('some email', FakeProvider([fenced]))
    assert result.ok
    assert result.attempts == 1

def test_nulls_are_allowed_for_missing_facts():
    sparse = dict(GOOD, customer_name=None, order_id=None, amount_usd=None, requested_action=None)
    result = extract('some email', FakeProvider([json.dumps(sparse)]))
    assert result.ok
    assert result.data.order_id is None

def test_missing_key_is_rejected():
    broken = dict(GOOD)
    del broken['summary']
    provider = FakeProvider([json.dumps(broken), json.dumps(GOOD)])
    result = extract('some email', provider)
    assert result.attempts == 2
    assert 'summary' in result.errors[0]

def test_empty_input_makes_zero_model_calls():
    provider = FakeProvider([json.dumps(GOOD)])
    result = extract('   \n', provider)
    assert not result.ok
    assert provider.calls == 0

def test_api_schema_drops_keys_the_api_does_not_accept():
    text = json.dumps(to_api_schema(TicketExtraction))
    for banned in ['pattern', 'minLength', 'maxLength', 'default']:
        assert banned not in text
    assert 'enum' in text

def test_token_counts_add_up_across_attempts():
    bad = dict(GOOD, order_id='20417')
    provider = FakeProvider([json.dumps(bad), json.dumps(GOOD)])
    result = extract('some email', provider)
    assert result.input_tokens > 0
    assert result.output_tokens > 0
