import pytest
from schemas import Brief
from scoping import Feature, pick_features
from pricing import price_scope, run_cost
from gate import run_gate
from evidence import check_numbers
def make_brief(**changes):
    base = dict(client='Lakeview Appliances', stated_problem='AI chatbot', underlying_problem='First reply takes 3 days', current_process='Agents copy order numbers by hand', volume_per_month=4200, minutes_per_task=9, cost_per_hour_usd=6, automatable_share=0.62, baseline_metric='3 days first reply', success_metric='first reply under 1 hour', data_sources=['order database'], data_access='needs_work', owner='Rahul', budget_usd=5000)
    base.update(changes)
    return Brief(**base)
def test_knapsack_beats_greedy():
    items = [Feature('a', 60, 10, 10, 10, False), Feature('b', 100, 20, 20, 20, False), Feature('c', 120, 30, 30, 30, False)]
    chosen, later = pick_features(items, 50)
    assert sorted(f.name for f in chosen) == ['b', 'c']
    assert [f.name for f in later] == ['a']
def test_required_feature_is_always_in():
    items = [Feature('must', 1, 5, 5, 5, True), Feature('nice', 9, 5, 5, 5, False)]
    chosen, later = pick_features(items, 5)
    assert [f.name for f in chosen] == ['must']
def test_required_over_budget_raises():
    with pytest.raises(ValueError):
        pick_features([Feature('must', 1, 5, 9, 12, True)], 5)
def test_quote_is_above_expected_price():
    hours, prices = price_scope([Feature('a', 5, 10, 20, 40, False)])
    assert prices['expected'] < prices['p80'] < prices['p95']
def test_gate_blocks_without_owner():
    result = run_gate(make_brief(owner=None), 4500, 150)
    assert result['verdict'] == 'NO GO FOR NOW'
def test_gate_blocks_slow_payback():
    result = run_gate(make_brief(volume_per_month=100), 4500, 5)
    assert result['verdict'] == 'NO GO FOR NOW'
def test_gate_goes_for_a_healthy_brief():
    result = run_gate(make_brief(), 4500, 150)
    assert result['verdict'] == 'GO'
    assert result['payback_months'] < 12
def test_number_check_catches_invented_number():
    assert check_numbers(make_brief(), 'We get 4200 tickets, 9 minutes each, 6 dollars, 62 percent, budget 5000') == []
    assert check_numbers(make_brief(volume_per_month=9100), 'We get 4200 tickets') != []
def test_run_cost_numbers():
    running = run_cost(4200, 0.62)
    assert running['tasks_per_month'] == 2604
