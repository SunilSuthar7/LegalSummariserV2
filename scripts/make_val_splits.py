import json
import random

DATA_PATH = "data/train_dataset.json"
VAL_PATH = "data/val_dataset.json"

random.seed(42)

data = json.load(open(DATA_PATH, encoding="utf-8"))
val = random.sample(data, 30)

with open(VAL_PATH, "w", encoding="utf-8") as f:
    json.dump(val, f, indent=2, ensure_ascii=False)

print(f"Saved {len(val)} validation samples")
