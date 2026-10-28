import os

from fastembed import TextEmbedding

name = os.getenv('EMBED_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
model = TextEmbedding(model_name=name)
vector = list(model.embed(['warm up']))[0]
print('model cached:', name, 'dimensions:', len(vector))
