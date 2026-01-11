import json
import torch
import argparse
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import sys, os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LEGALBERT_PATH = PROJECT_ROOT / "finetuned_legalbert_classifier"



def split_into_sentences(text):
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if len(s.strip()) > 20]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ratio", type=float, default=0.6)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(LEGALBERT_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(
        LEGALBERT_PATH
    ).to(device)
    model.eval()

    with open(args.input, encoding="utf-8") as f:
        data = json.load(f)

    results = []

    for sample in tqdm(data, desc="LegalBERT extractive"):
        sents = split_into_sentences(sample["text"])
        if not sents:
            continue

        enc = tokenizer(
            sents, padding=True, truncation=True,
            max_length=128, return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            probs = torch.softmax(model(**enc).logits, dim=1)[:, 1]

        ranked = sorted(zip(sents, probs), key=lambda x: x[1], reverse=True)
        keep = max(1, int(len(ranked) * args.ratio))

        extracted = " ".join(s for s, _ in ranked[:keep])

        results.append({
            "id": sample["id"],
            "text": extracted
        })

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"LegalBERT output saved to {args.output}")

if __name__ == "__main__":
    main()
