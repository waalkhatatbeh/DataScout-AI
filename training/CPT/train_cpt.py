
import json
import random
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.optim import AdamW


BASE_DIR = Path(__file__).resolve().parent

# Load configuration
with open(BASE_DIR / "config.yaml", "r") as f:
    config = yaml.safe_load(f)

MODEL_NAME = config["model"]["name"]
BLOCK_SIZE = config["model"]["block_size"]
EPOCHS = config["training"]["epochs"]
LR = config["training"]["learning_rate"]
BATCH_SIZE = config["training"]["batch_size"]
ACCUMULATION = config["training"]["gradient_accumulation_steps"]
SEED = config["training"]["seed"]

# Reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Prepare token blocks
def create_blocks(csv_path):
    df = pd.read_csv(csv_path)
    all_tokens = []

    for text in df["text"].dropna():
        ids = tokenizer.encode(
            str(text),
            add_special_tokens=False,
            verbose=False
        )
        all_tokens.extend(ids)
        all_tokens.append(tokenizer.eos_token_id)

    blocks = []

    for i in range(
        0,
        len(all_tokens) - BLOCK_SIZE + 1,
        BLOCK_SIZE
    ):
        blocks.append(all_tokens[i:i + BLOCK_SIZE])

    return blocks


class CPTDataset(Dataset):
    def __init__(self, blocks):
        self.blocks = blocks

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, idx):
        ids = torch.tensor(
            self.blocks[idx],
            dtype=torch.long
        )
        return {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids)
        }


train_blocks = create_blocks(
    BASE_DIR / config["data"]["train_file"]
)

val_blocks = create_blocks(
    BASE_DIR / config["data"]["validation_file"]
)

train_dataset = CPTDataset(train_blocks)
val_dataset = CPTDataset(val_blocks)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("Training blocks:", len(train_dataset))
print("Validation blocks:", len(val_dataset))

# Load GPT-2
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME
).to(device)

optimizer = AdamW(
    model.parameters(),
    lr=LR
)


def evaluate():
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in val_loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)

            outputs = model(
                input_ids=ids,
                attention_mask=mask,
                labels=ids
            )

            tokens = ids.numel() - ids.size(0)
            total_loss += outputs.loss.item() * tokens
            total_tokens += tokens

    loss = total_loss / total_tokens
    return loss, math.exp(loss)


# Training
history = []
best_val_loss = float("inf")

for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()

    total_loss = 0.0
    total_tokens = 0

    for step, batch in enumerate(train_loader):
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)

        outputs = model(
            input_ids=ids,
            attention_mask=mask,
            labels=ids
        )

        loss = outputs.loss
        (loss / ACCUMULATION).backward()

        tokens = ids.numel() - ids.size(0)
        total_loss += loss.item() * tokens
        total_tokens += tokens

        if (
            (step + 1) % ACCUMULATION == 0
            or (step + 1) == len(train_loader)
        ):
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )
            optimizer.step()
            optimizer.zero_grad()

    train_loss = total_loss / total_tokens
    val_loss, val_ppl = evaluate()

    result = {
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "validation_loss": val_loss,
        "validation_perplexity": val_ppl
    }

    history.append(result)

    print(f"\nEpoch {epoch + 1}/{EPOCHS}")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Perplexity: {val_ppl:.4f}")

    checkpoint = BASE_DIR / f"checkpoint_epoch_{epoch + 1}"
    checkpoint.mkdir(exist_ok=True)

    model.save_pretrained(checkpoint)
    tokenizer.save_pretrained(checkpoint)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        model.save_pretrained(BASE_DIR / "best_checkpoint")
        tokenizer.save_pretrained(BASE_DIR / "best_checkpoint")

with open(BASE_DIR / "training_history.json", "w") as f:
    json.dump(history, f, indent=2)

print("\nTraining completed!")
print("Best validation loss:", best_val_loss)
