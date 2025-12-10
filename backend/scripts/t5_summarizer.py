import json
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from tqdm import tqdm
import argparse
from pathlib import Path

# === Resolve folders relative to this file ===
# backend/scripts/ -> backend/
BACKEND_DIR = Path(__file__).resolve().parents[1]
# data/ is OUTSIDE backend/ (sibling of backend/)
DATA_DIR = BACKEND_DIR.parent / "data"
# sessions/ is INSIDE backend/
SESSIONS_DIR = BACKEND_DIR / "sessions"

# Configs
MODEL_NAME = "t5-base"
MAX_INPUT_TOKENS = 512
FINAL_SUM_MAX = 400
FINAL_MIN_LEN = 200
NUM_BEAMS = 8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
model.eval()

def summarize_text(text, min_len=FINAL_MIN_LEN, max_len=FINAL_SUM_MAX):
    input_enc = tokenizer.encode(
        "summarize: " + text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_TOKENS
    ).to(device)

    with torch.no_grad():
        summary_ids = model.generate(
            input_enc,
            max_length=max_len,
            min_length=min_len,
            num_beams=NUM_BEAMS
        )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def join_chunks(chunks):
    return " ".join(chunks)

def summarize_entry(entry):
    """Summarize one entry depending on its structure (chunked or plain)."""
    eid = entry.get("id")

    if "chunks" in entry:
        # Stage 1: summarize each chunk
        chunk_summaries = []
        for chunk in entry["chunks"]:
            chunk_summaries.append(
                summarize_text(chunk, min_len=100, max_len=250)
            )
        # Stage 2: combine chunk summaries and summarize again
        full_text = join_chunks(chunk_summaries)
        final_summary = summarize_text(full_text, min_len=FINAL_MIN_LEN, max_len=FINAL_SUM_MAX)
    else:
        # For non-chunked input 
        full_text = entry.get("input_text", "")
        final_summary = summarize_text(full_text, min_len=FINAL_MIN_LEN, max_len=FINAL_SUM_MAX)

    return {"id": eid, "refined_summary_improved": final_summary}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["ILC", "IN-ABS"], help="Choose dataset")
    parser.add_argument("--n", type=int, default=None, help="Number of entries to process")
    parser.add_argument("--ids", nargs="+", type=int, help="Specific entry IDs to process")
    parser.add_argument("--session_id", type=str, required=True, help="Unique session ID to save outputs separately")
    args = parser.parse_args()

    dataset = args.dataset.upper()

    # Input paths
    if dataset == "ILC":
        input_path = DATA_DIR / "chunked_ilc.json"
    else:
        input_path = DATA_DIR / "chunked_inabs.json"  # use chunked version

    # Session output folder
    session_dir = SESSIONS_DIR / args.session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    output_path = session_dir / (f"t5_{dataset.lower()}_final.json")

    print(f"Reading input from: {input_path}")
    print(f"Writing outputs to: {output_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # subset selection
    if args.ids:
        keep = set(args.ids)
        data = [d for d in data if d.get("id") in keep]
    elif args.n:
        data = data[:args.n]

    results = []
    for entry in tqdm(data, desc=f"Summarizing {dataset} entries"):
        results.append(summarize_entry(entry))

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f" Saved {len(results)} {dataset} summaries to {output_path}")
