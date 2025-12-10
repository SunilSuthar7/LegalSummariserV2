import json
import argparse
from rouge_score import rouge_scorer
from tqdm import tqdm
from pathlib import Path

# ===== CONFIG =====
DEFAULT_DATASET = "ILC"  # ILC or IN-ABS
BACKEND_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BACKEND_DIR.parent / "data"   # outside backend
# ==================

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET, help="Dataset name: ILC or IN-ABS")
parser.add_argument("--n", type=int, default=None, help="Number of entries to process")
parser.add_argument("--ids", nargs="+", type=int, help="Specific entry IDs to evaluate")
parser.add_argument("--session_id", type=str, required=True, help="Unique session ID for input/output paths")
args = parser.parse_args()

dataset = args.dataset.upper()
if dataset not in ["ILC", "IN-ABS"]:
    raise ValueError("Dataset must be 'ILC' or 'IN-ABS'")

# Paths
SESSIONS_DIR = BACKEND_DIR / "sessions"
session_dir = SESSIONS_DIR / args.session_id
session_dir.mkdir(parents=True, exist_ok=True)

extractive_path = session_dir / f"t5_{dataset.lower()}_final.json"

# Fix dataset filename mapping for reference summaries
if dataset == "ILC":
    reference_path = DATA_DIR / "cleaned_ilc.json"
elif dataset == "IN-ABS":
    reference_path = DATA_DIR / "cleaned_inabs.json"  # correct filename
else:
    raise ValueError(f"Unknown dataset: {dataset}")

output_path = session_dir / f"rouge_{dataset.lower()}.json"

# Load files
with open(extractive_path, "r", encoding="utf-8") as f:
    extractive_data = json.load(f)

with open(reference_path, "r", encoding="utf-8") as f:
    reference_data = json.load(f)

# Filter subset
if args.ids:
    extractive_data = [d for d in extractive_data if d["id"] in args.ids]
    reference_data = [d for d in reference_data if d["id"] in args.ids]
elif args.n:
    extractive_data = extractive_data[:args.n]
    reference_data = reference_data[:args.n]

# Build reference dict
reference_dict = {entry["id"]: entry["summary_text"] for entry in reference_data if "summary_text" in entry}

# Rouge scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
scores = {"rouge1": [], "rouge2": [], "rougeL": []}

processed = skipped_no_candidate = skipped_no_ref = 0

for entry in tqdm(extractive_data, desc="Evaluating"):
    eid = entry["id"]
    extractive_summary = entry.get("refined_summary_improved", "")
    reference_summary = reference_dict.get(eid, "")

    if not extractive_summary:
        skipped_no_candidate += 1
        continue
    if not reference_summary:
        skipped_no_ref += 1
        continue

    score = scorer.score(reference_summary, extractive_summary)
    for key in scores:
        scores[key].append(score[key].fmeasure)
    processed += 1

# Calculate averages
avg_scores = {k: (sum(v) / len(v) if v else 0) for k, v in scores.items()}

# Save results
with open(output_path, "w", encoding="utf-8") as f:
    json.dump({
        "processed": processed,
        "skipped_no_candidate": skipped_no_candidate,
        "skipped_no_reference": skipped_no_ref,
        "scores": avg_scores
    }, f, indent=2, ensure_ascii=False)

print(f"Evaluation complete! Results saved to {output_path}")
print(f"Processed entries: {processed}, Skipped candidate missing: {skipped_no_candidate}, Skipped ref missing: {skipped_no_ref}")
print("Average ROUGE Scores:")
for k, v in avg_scores.items():
    print(f"{k}: {v*100:.2f}%")
