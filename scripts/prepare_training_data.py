import json

# Load cleaned datasets
with open("data/cleaned_ilc.json", encoding="utf-8") as f:
    ilc = json.load(f)

with open("data/cleaned_inabs.json", encoding="utf-8") as f:
    inabs = json.load(f)

output = []

# -----------------------------
# Process ILC
# -----------------------------
for idx, sample in enumerate(ilc):
    output.append({
        "uid": f"ilc_{idx:06d}",
        "source": "ilc",
        "text": sample["input_text"],
        "summary": sample["summary_text"]
    })

# -----------------------------
# Process IN-ABS
# -----------------------------
for idx, sample in enumerate(inabs):
    output.append({
        "uid": f"inabs_{idx:06d}",
        "source": "inabs",
        "text": sample["input_text"],
        "summary": sample["summary_text"]
    })

# -----------------------------
# Save merged dataset
# -----------------------------
with open("data/train_dataset.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(
    f"Saved merged training dataset → data/train_dataset.json\n"
    f"ILC samples: {len(ilc)} | IN-ABS samples: {len(inabs)} | Total: {len(output)}"
)
