import pytest
import tools
from registry import ToolRegistry, registry
def test_real_tools_are_registered():
    assert registry.names() == ['calculate', 'get_stock_price']
def test_schema_is_built_from_signature_and_docstring():
    schema = {item['function']['name']: item for item in registry.schemas()}
    function = schema['get_stock_price']['function']
    assert function['parameters']['required'] == ['ticker']
    assert function['parameters']['properties']['ticker']['type'] == 'string'
    assert 'ticker symbol' in function['description']
    assert 'AAPL' in function['parameters']['properties']['ticker']['description']
def test_execute_success():
    result = registry.execute('calculate', expression='2 + 2')
    assert result.ok and result.value == 4 and result.content == '4'
def test_unknown_tool_gets_a_hint():
    result = registry.execute('calculat', expression='2 + 2')
    assert not result.ok
    assert 'Did you mean calculate' in result.content
def test_dangerous_tool_name_is_refused():
    result = registry.execute('delete_all_files', path='/')
    assert not result.ok and 'Unknown tool' in result.content
def test_extra_and_missing_arguments_are_refused():
    assert not registry.execute('calculate', expression='1', mode='x').ok
    assert not registry.execute('calculate').ok
def test_wrong_type_is_refused():
    result = registry.execute('calculate', expression=5)
    assert not result.ok and 'must be string' in result.content
def test_exception_inside_tool_becomes_a_result():
    result = registry.execute('calculate', expression='1 / 0')
    assert not result.ok and 'ZeroDivisionError' in result.content
def test_malicious_expression_becomes_a_result():
    result = registry.execute('calculate', expression='__import__(\'os\').getcwd()')
    assert not result.ok and result.content.startswith('Error:')
def test_bad_ticker_is_refused():
    assert not registry.execute('get_stock_price', ticker='AAPL; DROP TABLE').ok
    assert not registry.execute('get_stock_price', ticker='ZZZZ').ok
    assert registry.execute('get_stock_price', ticker=' aapl ').value == 227.52
def test_tool_name_can_be_an_argument_name():
    result = registry.execute('calculate', tool_name='x')
    assert not result.ok
def test_duplicate_and_missing_docstring_are_refused():
    local = ToolRegistry()
    @local.register
    def ping():
        '''Say pong.'''
        return 'pong'
    with pytest.raises(ValueError):
        local.register(ping)
    def silent():
        return 1
    with pytest.raises(ValueError):
        local.register(silent)
def test_approval_gate():
    seen = []
    def approver(name, arguments):
        seen.append((name, arguments))
        return arguments['amount'] <= 100
    local = ToolRegistry(approver=approver)
    @local.register(requires_approval=True)
    def refund(amount: int) -> str:
        '''Refund money to a customer.
        Args:
            amount: Dollars to refund
        '''
        return f'refunded {amount}'
    assert local.execute('refund', amount=50).ok
    assert not local.execute('refund', amount=500).ok
    assert len(seen) == 2
    assert not ToolRegistry().execute('refund').ok
