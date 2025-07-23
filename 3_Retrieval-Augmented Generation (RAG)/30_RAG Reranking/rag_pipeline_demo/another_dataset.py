from datasets import load_dataset

dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

with open("dataset_wiki.txt", "w") as f:
    f.write("\n".join(dataset[:4]["text"]))