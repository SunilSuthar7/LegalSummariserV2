import json
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from peft import PeftModel

MODEL_NAME = "t5-base"
ADAPTER_PATH = "finetuned_t5_qlora"  # set to None for base model
VAL_PATH = "data/val_dataset.json"

MAX_INPUT_LENGTH = 384
MAX_TARGET_LENGTH = 196

device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# Load tokenizer
# -----------------------------
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)

# -----------------------------
# Load base model
# -----------------------------
model = T5ForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)

# -----------------------------
# Load QLoRA adapter ONLY if provided
# -----------------------------
if ADAPTER_PATH is not None:
    print(f"Loading QLoRA adapter from: {ADAPTER_PATH}")
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
else:
    print("Running inference with BASE T5 (no adapter)")

model.eval()

# -----------------------------
# Load validation data
# -----------------------------
with open(VAL_PATH, encoding="utf-8") as f:
    data = json.load(f)

preds = []

# -----------------------------
# Inference loop
# -----------------------------
for sample in data:
    inputs = tokenizer(
        sample["text"],
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_LENGTH
    ).to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_TARGET_LENGTH,
            num_beams=4,
            early_stopping=True
        )

    pred = tokenizer.decode(output[0], skip_special_tokens=True)

    preds.append({
        "uid": sample.get("uid", ""),
        "reference": sample["summary"],
        "prediction": pred
    })

# -----------------------------
# Save predictions
# -----------------------------
out_path = "preds_qlora.json" if ADAPTER_PATH else "preds_base.json"

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(preds, f, indent=2, ensure_ascii=False)

print(f"Saved predictions → {out_path}")
