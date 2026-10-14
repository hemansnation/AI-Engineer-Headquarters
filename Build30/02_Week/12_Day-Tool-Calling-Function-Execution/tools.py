import re
from registry import register_tool
from safe_math import safe_calculate
STOCK_PRICES = {
    'AAPL': 227.52,
    'MSFT': 415.10,
    'GOOGL': 168.33,
    'AMZN': 186.40,
    'NVDA': 121.88,
    'TSLA': 248.25,
}
TICKER_PATTERN = re.compile(r'^[A-Z]{1,5}$')
@register_tool
def calculate(expression: str) -> float:
    '''Evaluate a math expression and return the exact number. Use this for every calculation, even small ones.
    Args:
        expression: Math written with numbers, parentheses and + - * / // % ** for example (15 / 100) * 2480 + 37 * 12
    '''
    result = safe_calculate(expression)
    if isinstance(result, float):
        result = round(result, 10)
    return result
@register_tool
def get_stock_price(ticker: str) -> float:
    '''Get the latest sample price in US dollars for one stock ticker symbol.
    Args:
        ticker: Stock ticker in capital letters, for example AAPL
    '''
    symbol = ticker.strip().upper()
    if not TICKER_PATTERN.match(symbol):
        raise ValueError('A ticker is one to five letters, for example AAPL.')
    if symbol not in STOCK_PRICES:
        raise ValueError(f'No price for {symbol}. Known tickers: {sorted(STOCK_PRICES)}.')
    return STOCK_PRICES[symbol]
