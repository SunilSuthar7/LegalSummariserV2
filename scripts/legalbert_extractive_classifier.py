import json
import torch
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

INPUT_PATH = "data/val_dataset.json"
OUTPUT_PATH = "data/val_extractive_legalbert_classifier.json"

MODEL_PATH = "finetuned_legalbert_classifier"
EXTRACTIVE_RATIO = 0.6

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(device)
model.eval()

def split_into_sentences(text):
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sents if len(s.strip()) > 20]

def extractive_summary(text):
    sents = split_into_sentences(text)
    if not sents:
        return ""

    enc = tokenizer(
        sents,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().tolist()

    ranked = sorted(
        zip(sents, probs),
        key=lambda x: x[1],
        reverse=True
    )

    keep = int(len(ranked) * EXTRACTIVE_RATIO)
    selected = [s for s, _ in ranked[:keep]]

    return " ".join(selected)

with open(INPUT_PATH, encoding="utf-8") as f:
    data = json.load(f)

results = []

for sample in tqdm(data, desc="Classifier-based extractive"):
    ext = extractive_summary(sample["text"])
    results.append({
        "uid": sample.get("uid", ""),
        "text": ext,
        "summary": sample["summary"]
    })

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"Saved classifier-based extractive output → {OUTPUT_PATH}")
