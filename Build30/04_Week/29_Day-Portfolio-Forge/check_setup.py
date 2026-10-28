import shutil
import subprocess
import sys
import ollama
from config import EMBED_MODEL, OLLAMA_MODEL
def check_git() -> None:
    print('git:', subprocess.run(['git', '--version'], capture_output=True, text=True).stdout.strip())
def check_python() -> None:
    print('python:', sys.version.split()[0])
def check_ollama() -> None:
    if not shutil.which('ollama'):
        print('ollama: not installed, get it from ollama.com')
        return
    names = [item.model for item in ollama.list().models]
    print('ollama models:', names)
    print('chat model ready:', any(name.startswith(OLLAMA_MODEL) for name in names))
def check_embeddings() -> None:
    from embed import encode
    vectors = encode(['hello portfolio'])
    print('embedding model:', EMBED_MODEL, 'shape:', vectors.shape)
if __name__ == '__main__':
    check_python()
    check_git()
    check_ollama()
    check_embeddings()
