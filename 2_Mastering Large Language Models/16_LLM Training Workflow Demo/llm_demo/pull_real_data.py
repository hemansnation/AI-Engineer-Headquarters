from datasets import load_dataset

dataset = load_dataset("bigcode/the-stack", split="train", streaming=True)

with open("data/raw/stack_subset.json", "w") as f:
    for i, sample in enumerate(dataset.take(10)):
        f.write(str(sample) + "\n")
