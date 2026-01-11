import sys
import os

# ✅ ADD PROJECT ROOT TO PYTHON PATH
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

import json
import argparse
from pathlib import Path
from src.cleaner import clean_text

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)

    cleaned = []
    for item in data:
        text = clean_text(item["input_text"], aggressive=False)
        if text.strip():
            cleaned.append({
                "id": item["id"],
                "text": text
            })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, indent=2, ensure_ascii=False)

    print(f"Cleaned data saved to {output_path}")

if __name__ == "__main__":
    main()
