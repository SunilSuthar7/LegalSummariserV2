import json
from rouge_score import rouge_scorer

# ======================================
# CONFIG
# ======================================
PRED_PATH = "data/t5_val_predictions.json"   

# ======================================
# LOAD DATA
# ======================================
with open(PRED_PATH, encoding="utf-8") as f:
    data = json.load(f)

print(f"Loaded {len(data)} predictions")

# ======================================
# ROUGE SETUP
# ======================================
scorer = rouge_scorer.RougeScorer(
    ["rouge1", "rouge2", "rougeL"],
    use_stemmer=True
)

scores = {
    "rouge1": [],
    "rouge2": [],
    "rougeL": []
}

# ======================================
# SCORE LOOP
# ======================================
for item in data:
    reference = item["reference"]
    prediction = item["prediction"]

    if not reference or not prediction:
        continue

    s = scorer.score(reference, prediction)
    scores["rouge1"].append(s["rouge1"].fmeasure)
    scores["rouge2"].append(s["rouge2"].fmeasure)
    scores["rougeL"].append(s["rougeL"].fmeasure)

# ======================================
# RESULTS
# ======================================
print("\nROUGE F1 scores:")
for k in scores:
    avg = sum(scores[k]) / len(scores[k]) if scores[k] else 0
    print(f"{k}: {avg * 100:.2f}%")

