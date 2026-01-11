import json
import re
from rouge_score import rouge_scorer
from tqdm import tqdm

INPUT_PATH = "data/train_dataset.json"   # same file used for T5
OUTPUT_PATH = "data/legalbert_sentence_data.json"

scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)

def split_into_sentences(text):
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sents if len(s.strip()) > 20]

dataset = []

with open(INPUT_PATH, encoding="utf-8") as f:
    data = json.load(f)

for sample in tqdm(data, desc="Building sentence dataset"):
    text = sample["text"]
    summary = sample["summary"]

    sentences = split_into_sentences(text)

    for sent in sentences:
        score = scorer.score(summary, sent)["rouge1"].fmeasure
        label = 1 if score >= 0.25 else 0   # threshold used in literature

        dataset.append({
            "sentence": sent,
            "label": label
        })

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(dataset, f, indent=2, ensure_ascii=False)

print(f"Saved sentence classification dataset → {OUTPUT_PATH}")
