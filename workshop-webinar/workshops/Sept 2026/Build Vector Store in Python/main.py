from sentence_transformers import SentenceTransformer
import math

def dot_product(vector_a, vector_b):
    total = 0.0
    for a, b in zip(vector_a, vector_b):
        total += a * b
    return total

def magnitude(vector):
    total = 0.0
    for value in vector:
        total += value ** 2
    return math.sqrt(total)

def cosine_similarity(vector_a, vector_b):
    dot = dot_product(vector_a, vector_b)
    mag_a = magnitude(vector_a)
    mag_b = magnitude(vector_b)

    if mag_a == 0 or mag_b == 0:
        return 0.0

    return dot / (mag_a * mag_b)


class VectorStore:
    def __init__(self):
        self.items = []

    def add(self, text, vector):
        self.items.append({
            "text": text,
            "vector": list(vector)
        })

    def query(self, query_vector, top_k=2):
        scored_items = []

        for item in self.items:
            score = cosine_similarity(query_vector, item["vector"])
            scored_items.append((score, item["text"]))

        scored_items.sort(key=lambda pair: pair[0], reverse=True)
        return scored_items[:top_k]


model = SentenceTransformer('all-MiniLM-L6-v2')

sentences = [
    "The cat sat on the mat.",
    "The dog is playing in the park.",
    "The stock market crashed today.",
    "Investors are worried about inflation.",
    "The kitten is sleeping on the sofa."
]

embeddings = model.encode(sentences)

for sentence, vector in zip(sentences, embeddings):
    print(sentence)
    print("Vector length:", len(vector))
    print("First 5 numbers of the vector:", vector[:5])
    print("------")

vector_as_list = list(vector)

# example test
# a = [1,0]
# b = [1,0]
# c = [0,1]

# print(cosine_similarity(a, b))  # 1.0
# print(cosine_similarity(a, c))  # 0.0


# 1.0 they point in same direction
# 0.0 the are unrelated or at a right angle
# -1.0 they point in opposite directions

store = VectorStore()

for sentence in sentences:
    vector = model.encode(sentence)
    store.add(sentence, vector)

# query_text = "A kitten is playing with a toy."
query_text = "How the economy doing?"
query_vector = model.encode(query_text)

results = store.query(query_vector, top_k=2)

print(f"Query: {query_text}")
print("Top 2 results:")
for score, text in results:
    print(f"Score: {score:.4f}, Text: {text}")