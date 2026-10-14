import inspect
import time
from dataclasses import dataclass
from difflib import get_close_matches
from typing import Any, Callable
TYPE_NAMES = {str: 'string', int: 'integer', float: 'number', bool: 'boolean'}
MAX_TEXT_LENGTH = 500
@dataclass
class ToolSpec:
    name: str
    func: Callable
    schema: dict
    requires_approval: bool = False
@dataclass
class ToolResult:
    name: str
    arguments: dict
    ok: bool
    value: Any
    content: str
    seconds: float
def build_schema(func):
    doc = inspect.getdoc(func) or ''
    summary, _, args_block = doc.partition('Args:')
    descriptions = {}
    for line in args_block.splitlines():
        key, _, text = line.strip().partition(':')
        if text:
            descriptions[key.strip()] = text.strip()
    properties = {}
    required = []
    for name, param in inspect.signature(func).parameters.items():
        annotation = param.annotation if param.annotation is not inspect.Parameter.empty else str
        properties[name] = {
            'type': TYPE_NAMES.get(annotation, 'string'),
            'description': descriptions.get(name, ''),
        }
        if param.default is inspect.Parameter.empty:
            required.append(name)
    return {
        'type': 'function',
        'function': {
            'name': func.__name__,
            'description': ' '.join(summary.split()),
            'parameters': {'type': 'object', 'properties': properties, 'required': required},
        },
    }
def type_problem(name, annotation, value):
    if annotation is inspect.Parameter.empty or annotation not in TYPE_NAMES:
        return None
    if annotation is float:
        good = isinstance(value, (int, float)) and not isinstance(value, bool)
    elif annotation is int:
        good = isinstance(value, int) and not isinstance(value, bool)
    else:
        good = isinstance(value, annotation)
    if not good:
        return f'{name} must be {TYPE_NAMES[annotation]}, but I received {type(value).__name__}.'
    if isinstance(value, str) and len(value) > MAX_TEXT_LENGTH:
        return f'{name} is longer than {MAX_TEXT_LENGTH} characters.'
    return None
class ToolRegistry:
    def __init__(self, approver=None):
        self._tools = {}
        self._approver = approver
    def register(self, func=None, *, requires_approval=False):
        def add(target):
            name = target.__name__
            if name in self._tools:
                raise ValueError(f'Tool {name} is already registered.')
            if not (inspect.getdoc(target) or '').strip():
                raise ValueError(f'Tool {name} needs a docstring. The model reads it.')
            self._tools[name] = ToolSpec(name, target, build_schema(target), requires_approval)
            return target
        if func is None:
            return add
        return add(func)
    def names(self):
        return sorted(self._tools)
    def schemas(self):
        return [spec.schema for spec in self._tools.values()]
    def execute(self, tool_name, /, **kwargs):
        started = time.perf_counter()
        spec = self._tools.get(tool_name)
        if spec is None:
            message = f'Unknown tool {tool_name}. Available tools: {self.names()}.'
            close = get_close_matches(str(tool_name), self._tools, n=1)
            if close:
                message += f' Did you mean {close[0]}?'
            return self._fail(tool_name, kwargs, message, started)
        signature = inspect.signature(spec.func)
        try:
            bound = signature.bind(**kwargs)
        except TypeError as error:
            return self._fail(tool_name, kwargs, f'Bad arguments for {tool_name}: {error}', started)
        for name, value in bound.arguments.items():
            problem = type_problem(name, signature.parameters[name].annotation, value)
            if problem:
                return self._fail(tool_name, kwargs, f'Bad arguments for {tool_name}: {problem}', started)
        if spec.requires_approval:
            approved = self._approver is not None and self._approver(tool_name, dict(bound.arguments))
            if not approved:
                return self._fail(tool_name, kwargs, f'{tool_name} needs human approval and did not get it.', started)
        try:
            value = spec.func(*bound.args, **bound.kwargs)
        except Exception as error:
            return self._fail(tool_name, kwargs, f'{tool_name} failed with {type(error).__name__}: {error}', started)
        return ToolResult(tool_name, dict(kwargs), True, value, str(value), time.perf_counter() - started)
    def _fail(self, tool_name, kwargs, message, started):
        return ToolResult(str(tool_name), dict(kwargs), False, None, f'Error: {message}', time.perf_counter() - started)
registry = ToolRegistry()
register_tool = registry.register
