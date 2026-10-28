import os
from pathlib import Path
from sandbox import run_python
secret_file = Path('decoy_secret.txt').resolve()
secret_file.write_text('pretend-api-key-123', encoding='utf-8')
os.environ['FAKE_API_KEY'] = 'pretend-env-secret'
generated_code = f'''import os
print(open({str(secret_file)!r}).read())
print(os.environ.get('FAKE_API_KEY'))
'''
print('--- naive exec inside your own process ---')
exec(generated_code)
print('--- same code inside the sandbox ---')
print(run_python(generated_code, backend='local').for_model())
secret_file.unlink()
