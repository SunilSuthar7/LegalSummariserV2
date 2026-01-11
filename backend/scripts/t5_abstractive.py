import json
import argparse
import torch
import re
from transformers import T5ForConditionalGeneration, T5Tokenizer
from peft import PeftModel
from tqdm import tqdm
import sys
from pathlib import Path
from typing import List

# ================= PATHS =================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

T5_BASE_NAME = "t5-base"
T5_ADAPTER_PATH = PROJECT_ROOT / "finetuned_t5_qlora"

# ================= CONFIG (SAFE DEFAULTS) =================
MAX_INPUT_TOKENS = 512          # tokenizer truncation cap
CHUNK_TOKENS = 300              # CPU/GPU safe
STAGE1_MAX = 120
STAGE1_MIN = 60
FINAL_MAX = 350
FINAL_MIN = 150
NUM_BEAMS = 4                   # CPU-friendly
KEYWORD_SENT_LIMIT = 5
# =========================================================

KEYWORDS = [
    "section", "sections", "appeal", "judgment", "petition",
    "supreme court", "tribunal", "settlement", "mediation",
    "arbitration", "inherent power", "Full Bench"
]

# ================= HELPERS =================

def split_into_sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]

def find_keyword_sentences(text: str, limit: int) -> List[str]:
    sents = split_into_sentences(text)
    found = []
    lowered = [s.lower() for s in sents]
    for kw in KEYWORDS:
        for i, s in enumerate(lowered):
            if kw in s and sents[i] not in found:
                found.append(sents[i])
                if len(found) >= limit:
                    return found
    return found

def chunk_text_by_tokens(text: str, tokenizer, max_tokens: int) -> List[str]:
    words = text.split()
    chunks, current = [], []
    for w in words:
        current.append(w)
        if len(tokenizer.encode(" ".join(current))) >= max_tokens:
            chunks.append(" ".join(current))
            current = []
    if current:
        chunks.append(" ".join(current))
    return chunks

def summarize_text(
    text: str,
    tokenizer,
    model,
    max_length: int,
    min_length: int,
    device
) -> str:
    enc = tokenizer(
        "summarize: " + text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_TOKENS
    ).to(device)

    with torch.no_grad():
        out = model.generate(
            **enc,
            max_length=max_length,
            min_length=min_length,
            num_beams=NUM_BEAMS,
            no_repeat_ngram_size=3,
            early_stopping=True
        )

    return tokenizer.decode(out[0], skip_special_tokens=True)

# ================= MAIN =================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = T5Tokenizer.from_pretrained(T5_BASE_NAME)

    base_model = T5ForConditionalGeneration.from_pretrained(
        T5_BASE_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    ).to(device)

    model = PeftModel.from_pretrained(base_model, T5_ADAPTER_PATH)
    model.eval()

    with open(args.input, encoding="utf-8") as f:
        data = json.load(f)

    results = []

    for sample in tqdm(data, desc="T5 hierarchical summarization"):
        text = sample["text"].strip()
        if not text:
            continue

        # -------- Stage 1: Chunk-wise summaries --------
        chunks = chunk_text_by_tokens(text, tokenizer, CHUNK_TOKENS)

        stage1_summaries = []
        for chunk in chunks:
            s = summarize_text(
                chunk,
                tokenizer,
                model,
                STAGE1_MAX,
                STAGE1_MIN,
                device
            )
            stage1_summaries.append(s)

        # -------- Stage 2: Optional merge --------
        if len(stage1_summaries) <= 2:
            final_summary = " ".join(stage1_summaries)
        else:
            combined = " ".join(stage1_summaries)
            final_summary = summarize_text(
                combined,
                tokenizer,
                model,
                FINAL_MAX,
                FINAL_MIN,
                device
            )

        # -------- Keyword reinforcement (cheap & effective) --------
        keywords = find_keyword_sentences(text, KEYWORD_SENT_LIMIT)
        prepend = [k for k in keywords if k not in final_summary]
        if prepend:
            final_summary = " ".join(prepend) + " " + final_summary

        final_summary = re.sub(r"\s+", " ", final_summary).strip()

        results.append({
            "id": sample.get("id"),
            "summary_text": final_summary
        })

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"T5 summaries saved to {args.output}")

if __name__ == "__main__":
    main()
