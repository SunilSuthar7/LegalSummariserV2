import json
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from peft import PeftModel
from typing import List
import re
from tqdm import tqdm
import time

# ==================================
# CONFIG
# ==================================
INPUT_PATH = "data/val_extractive_legalbert.json"
OUTPUT_PATH = "data/t5_val_predictions.json"

MODEL_NAME = "t5-base"
USE_QLORA = True
ADAPTER_PATH = "finetuned_t5_qlora"

MAX_INPUT_TOKENS = 512
CHUNK_SUM_MAX = 120          # slightly longer chunk summaries
FINAL_SUM_MAX = 360          # longer final summaries
FINAL_MIN_LEN = 120

NUM_BEAMS = 12               # ROUGE-biased decoding
LENGTH_PENALTY = 0.85
NO_REPEAT_NGRAM = 4

KEYWORD_SENT_LIMIT = 7
EXTRACTIVE_RATIO = 0.6       # 60% extractive bias

BATCH_SIZE = 4
SLEEP_BETWEEN_BATCHES = 1
# ==================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)

# ==================================
# LOAD MODEL
# ==================================
base_model = T5ForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16
).to(device)

if USE_QLORA:
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
else:
    model = base_model

model.eval()

# ==================================
# LEGAL KEYWORDS (expanded)
# ==================================
KEYWORDS = [
    "held that", "it is held", "it is observed", "accordingly",
    "appeal is dismissed", "appeal is allowed", "petition is dismissed",
    "section", "sections", "FIR", "settlement", "arbitration",
    "inherent power", "supreme court", "high court", "tribunal",
    "judgment", "order"
]

# ==================================
# HELPERS
# ==================================
def split_into_sentences(text: str) -> List[str]:
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sents if s.strip()]

def score_sentence(sent: str) -> int:
    score = 0
    lower = sent.lower()
    for kw in KEYWORDS:
        if kw in lower:
            score += 3
    if re.search(r"\b(section|sec\.?)\s+\d+", lower):
        score += 2
    if len(sent.split()) > 20:
        score += 1
    return score

def extractive_filter(text: str, ratio: float) -> str:
    sents = split_into_sentences(text)
    scored = [(score_sentence(s), s) for s in sents]
    scored.sort(reverse=True, key=lambda x: x[0])

    keep = max(1, int(len(scored) * ratio))
    selected = [s for _, s in scored[:keep]]

    return " ".join(selected)

def summarize_batch(batch_texts: List[str], max_length: int, min_length: int):
    with torch.cuda.amp.autocast():
        enc = tokenizer(
            ["summarize: " + t for t in batch_texts],
            return_tensors="pt",
            max_length=MAX_INPUT_TOKENS,
            truncation=True,
            padding=True
        ).to(device)

        out = model.generate(
            input_ids=enc.input_ids,
            attention_mask=enc.attention_mask,
            max_length=max_length,
            min_length=min_length,
            num_beams=NUM_BEAMS,
            length_penalty=LENGTH_PENALTY,
            no_repeat_ngram_size=NO_REPEAT_NGRAM,
            early_stopping=True
        )

    torch.cuda.empty_cache()
    if SLEEP_BETWEEN_BATCHES:
        time.sleep(SLEEP_BETWEEN_BATCHES)

    return [tokenizer.decode(o, skip_special_tokens=True) for o in out]

def adaptive_group_chunks(text: str) -> List[str]:
    tokens = tokenizer.encode(text)
    groups = []
    for i in range(0, len(tokens), MAX_INPUT_TOKENS):
        chunk = tokenizer.decode(tokens[i:i+MAX_INPUT_TOKENS])
        groups.append(chunk)
    return groups

def two_stage_summarize(text: str) -> str:
    # METHOD 1: aggressive extractive filtering
    filtered = extractive_filter(text, EXTRACTIVE_RATIO)

    chunks = adaptive_group_chunks(filtered)

    stage1 = []
    for i in range(0, len(chunks), BATCH_SIZE):
        stage1.extend(summarize_batch(chunks[i:i+BATCH_SIZE], CHUNK_SUM_MAX, 40))

    combined = " ".join(stage1)
    final = summarize_batch([combined], FINAL_SUM_MAX, FINAL_MIN_LEN)[0]

    # Explicit judgment sentence injection
    judgment_sents = [
        s for s in split_into_sentences(text)
        if any(k in s.lower() for k in ["appeal", "petition", "dismissed", "allowed"])
    ]

    prepend = judgment_sents[:KEYWORD_SENT_LIMIT]
    if prepend:
        final = " ".join(prepend) + " " + final

    return re.sub(r"\s+", " ", final).strip()

# ==================================
# MAIN
# ==================================
with open(INPUT_PATH, encoding="utf-8") as f:
    data = json.load(f)

results = []

for sample in tqdm(data, desc="Summarizing validation set", ncols=100):
    summary = two_stage_summarize(sample["text"])

    results.append({
        "uid": sample.get("uid", ""),
        "reference": sample["summary"],
        "prediction": summary
    })

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\nSaved predictions → {OUTPUT_PATH}")
