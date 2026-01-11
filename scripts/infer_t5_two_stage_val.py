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
INPUT_PATH = "data/val_extractive_legalbert_classifier.json"
OUTPUT_PATH = "data/t5_val_predictions.json"

MODEL_NAME = "t5-base"
USE_QLORA = True          # True = QLoRA, False = base T5
ADAPTER_PATH = "finetuned_t5_qlora"

MAX_INPUT_TOKENS = 512
CHUNK_SUM_MAX = 100
FINAL_SUM_MAX = 300
FINAL_MIN_LEN = 90
NUM_BEAMS = 8
LENGTH_PENALTY = 1.0
KEYWORD_SENT_LIMIT = 5

BATCH_SIZE = 4
SLEEP_BETWEEN_BATCHES = 1
# ==================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)

# ==================================
# LOAD MODEL (BASE OR QLoRA)
# ==================================
base_model = T5ForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16
).to(device)

if USE_QLORA:
    print("Using QLoRA fine-tuned model")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
else:
    print("Using base T5 model")
    model = base_model

model.eval()

# ==================================
# KEYWORDS
# ==================================
KEYWORDS = [
    'mediation', 'conciliation', 'FIR', 'settlement', 'agreed',
    'section', 'sections', '498A', '323', '354', '504',
    'arbitration', 'settlement agreement', 'inherent power',
    'Full Bench', 'tribunal', 'appeal', 'supreme court',
    'judgment', 'petition'
]

# ==================================
# HELPERS
# ==================================
def split_into_sentences(text: str) -> List[str]:
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sents if s.strip()]

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
            early_stopping=True,
            no_repeat_ngram_size=3
        )

        summaries = [tokenizer.decode(o, skip_special_tokens=True) for o in out]

    torch.cuda.empty_cache()
    if SLEEP_BETWEEN_BATCHES:
        time.sleep(SLEEP_BETWEEN_BATCHES)

    return summaries

def adaptive_group_chunks(chunks: List[str]) -> List[str]:
    groups, current, tokens = [], [], 0
    for chunk in chunks:
        ct = len(tokenizer.encode(chunk))
        if tokens + ct > MAX_INPUT_TOKENS:
            groups.append(" ".join(current))
            current, tokens = [chunk], ct
        else:
            current.append(chunk)
            tokens += ct
    if current:
        groups.append(" ".join(current))
    return groups

def two_stage_summarize(text: str) -> str:
    chunks = adaptive_group_chunks([text])
    stage1 = []
    for i in range(0, len(chunks), BATCH_SIZE):
        stage1.extend(summarize_batch(chunks[i:i+BATCH_SIZE], CHUNK_SUM_MAX, 20))

    combined = " ".join(stage1)
    final = summarize_batch([combined], FINAL_SUM_MAX, FINAL_MIN_LEN)[0]

    keywords = find_keyword_sentences(text, KEYWORD_SENT_LIMIT)
    prepend = [k for k in keywords if k not in final]

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
