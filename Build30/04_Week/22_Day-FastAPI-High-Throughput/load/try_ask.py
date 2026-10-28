import httpx
response = httpx.post(
    'http://127.0.0.1:8000/ask',
    json={'question': 'Why does time.sleep freeze an async endpoint?'},
    headers={'x-api-key': 'me'},
    timeout=120,
)
print(response.status_code)
print(response.headers['x-request-id'])
print(response.json())
