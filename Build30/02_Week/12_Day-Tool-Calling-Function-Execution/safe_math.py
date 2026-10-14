import ast
import operator
BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
UNARY_OPERATORS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
MAX_LENGTH = 200
MAX_EXPONENT = 100
MAX_RESULT_BITS = 4096
class UnsafeExpressionError(ValueError):
    pass
def safe_calculate(expression):
    if not isinstance(expression, str):
        raise UnsafeExpressionError('Expression must be text.')
    if len(expression) > MAX_LENGTH:
        raise UnsafeExpressionError(f'Expression is longer than {MAX_LENGTH} characters.')
    try:
        tree = ast.parse(expression.strip(), mode='eval')
    except SyntaxError as error:
        raise UnsafeExpressionError('That is not a valid math expression.') from error
    return _evaluate(tree.body)
def _evaluate(node):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        raise UnsafeExpressionError('Only plain numbers are allowed.')
    if isinstance(node, ast.BinOp) and type(node.op) in BINARY_OPERATORS:
        left = _evaluate(node.left)
        right = _evaluate(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
            raise UnsafeExpressionError(f'Exponent is bigger than {MAX_EXPONENT}.')
        result = BINARY_OPERATORS[type(node.op)](left, right)
        if isinstance(result, int) and result.bit_length() > MAX_RESULT_BITS:
            raise UnsafeExpressionError('Result is too large.')
        return result
    if isinstance(node, ast.UnaryOp) and type(node.op) in UNARY_OPERATORS:
        return UNARY_OPERATORS[type(node.op)](_evaluate(node.operand))
    raise UnsafeExpressionError('Only numbers and + - * / // % ** with parentheses are allowed.')
