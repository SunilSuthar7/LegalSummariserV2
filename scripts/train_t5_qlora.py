# train_t5_qlora.py

import json
import torch
from datasets import Dataset
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)

MODEL_NAME = "t5-base"
TRAIN_PATH = "data/train_dataset.json"

MAX_INPUT_LENGTH = 384          # 512 WILL OOM
MAX_TARGET_LENGTH = 196

SAMPLE_SIZE = 100               # 100 → 500 → 2k → FULL
BATCH_SIZE = 1
GRAD_ACCUM = 16                 # Effective batch = 16

LR = 1e-4                       # Lower for QLoRA
EPOCHS = 3

OUTPUT_DIR = "finetuned_t5_qlora"

# =====================================================
# LOAD DATA
# =====================================================
print("\nLoading dataset...")
with open(TRAIN_PATH, encoding="utf-8") as f:
    data = json.load(f)

data = data[:SAMPLE_SIZE]
dataset = Dataset.from_list(data)

print(f"Loaded {len(dataset)} samples")

# =====================================================
# TOKENIZER
# =====================================================
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)

# =====================================================
# 4-BIT QLoRA CONFIG
# =====================================================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

print("\nLoading T5 model in 4-bit (QLoRA)...")

model = T5ForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto"
)

# =====================================================
# PREPARE MODEL FOR QLoRA TRAINING
# =====================================================
model = prepare_model_for_kbit_training(model)

model.gradient_checkpointing_enable()
model.config.use_cache = False

# =====================================================
# LoRA ADAPTER CONFIG (QLoRA)
# =====================================================
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q", "v"],
    bias="none",
    task_type="SEQ_2_SEQ_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# =====================================================
# TOKENIZATION
# =====================================================
def tokenize(batch):
    inputs = tokenizer(
        batch["text"],
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        padding="max_length"
    )

    targets = tokenizer(
        batch["summary"],
        max_length=MAX_TARGET_LENGTH,
        truncation=True,
        padding="max_length"
    )

    inputs["labels"] = targets["input_ids"]
    return inputs

print("\nTokenizing dataset...")
dataset = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
print("Tokenization done")

# =====================================================
# TRAINING SETUP
# =====================================================
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    num_train_epochs=EPOCHS,
    logging_steps=20,
    save_steps=500,
    save_total_limit=2,
    fp16=True,
    optim="paged_adamw_8bit",     # REQUIRED FOR QLoRA
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# =====================================================
# TRAIN
# =====================================================
print("\nStarting QLoRA training...\n")
trainer.train()

# =====================================================
# SAVE ADAPTER
# =====================================================
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"\nTraining complete. QLoRA adapter saved to: {OUTPUT_DIR}")
