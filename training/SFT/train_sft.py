
import os
import json
import inspect
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, TrainerCallback

HERE = Path(__file__).resolve().parent
CONFIG_FILE = HERE / "config.yaml"
settings = {}
if CONFIG_FILE.exists():
    for line in CONFIG_FILE.read_text(encoding="utf-8").splitlines():
        if ":" in line and not line.lstrip().startswith("#"):
            key, value = line.split(":", 1)
            settings[key.strip()] = value.strip().strip('"')
DATA_PATH = Path(os.environ.get("SFT_DATA", settings.get("sft_data", str(HERE / "sft_pairs.jsonl"))))
CPT_CHECKPOINT = os.environ.get(
    "CPT_CHECKPOINT",
    settings.get("cpt_checkpoint", "/kaggle/input/datasets/toqa06ds/datascout-ai-sft-300-training/checkpoint_epoch2/checkpoint_epoch2")
)
OUTPUT_DIR = Path(os.environ.get("SFT_OUTPUT", settings.get("sft_output", "/kaggle/working/datascout_sft_300_validated")))

MAX_LENGTH = int(os.environ.get("MAX_LENGTH", settings.get("max_length", "512")))
EPOCHS = int(os.environ.get("EPOCHS", settings.get("epochs", "3")))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", settings.get("batch_size", "4")))
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", settings.get("grad_accum", "4")))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", settings.get("learning_rate", "2e-5")))
SPECIAL_TOKENS = ["<|user|>", "<|assistant|>", "<|end|>"]

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Training file not found: {DATA_PATH}")
if not Path(CPT_CHECKPOINT).exists():
    raise FileNotFoundError(
        f"CPT checkpoint not found: {CPT_CHECKPOINT}\n"
        "Attach the CPT checkpoint as a Kaggle input, then set CPT_CHECKPOINT to its exact folder."
    )

tokenizer = AutoTokenizer.from_pretrained(CPT_CHECKPOINT, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(CPT_CHECKPOINT)

tokenizer.add_special_tokens(
    {"additional_special_tokens": SPECIAL_TOKENS}
)
if len(tokenizer) != model.get_input_embeddings().num_embeddings:
    model.resize_token_embeddings(len(tokenizer))

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
model.config.pad_token_id = tokenizer.pad_token_id
model.config.use_cache = False

tokenizer.chat_template = (
    "{% for message in messages %}"
    "{% if message['role'] == 'user' %}<|user|> {{ message['content'] }}\n"
    "{% elif message['role'] == 'assistant' %}<|assistant|> {{ message['content'] }}\n<|end|>\n"
    "{% endif %}{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|> {% endif %}"
)

def read_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            obj = json.loads(line)
            if not obj.get("question") or not obj.get("answer"):
                raise ValueError(f"Missing question/answer at JSONL line {line_no}")
            rows.append(obj)
    if not rows:
        raise ValueError("No training rows found.")
    return rows

class SFTDataset(Dataset):
    def __init__(self, rows):
        self.items = []
        self.masked_prompt_tokens = 0
        self.supervised_answer_tokens = 0
        self.max_seen_length = 0

        end_id = tokenizer.convert_tokens_to_ids("<|end|>")
        for row in rows:
            prompt = f"<|user|> {row['question']}\n<|assistant|>"
            answer = f" {row['answer']}\n<|end|>"

            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]

            if len(answer_ids) >= MAX_LENGTH:
                answer_ids = answer_ids[:MAX_LENGTH - 1] + [end_id]
                prompt_ids = []
            elif len(prompt_ids) + len(answer_ids) > MAX_LENGTH:
                prompt_ids = prompt_ids[-(MAX_LENGTH - len(answer_ids)):]

            input_ids = prompt_ids + answer_ids
            labels = ([-100] * len(prompt_ids)) + answer_ids.copy()
            attention_mask = [1] * len(input_ids)

            self.masked_prompt_tokens += len(prompt_ids)
            self.supervised_answer_tokens += len(answer_ids)
            self.max_seen_length = max(self.max_seen_length, len(input_ids))
            self.items.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels
            })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]

class AnswerOnlyCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        max_len = max(len(x["input_ids"]) for x in features)
        input_ids, attention_masks, labels = [], [], []
        for x in features:
            pad = max_len - len(x["input_ids"])
            input_ids.append(x["input_ids"] + [self.pad_token_id] * pad)
            attention_masks.append(x["attention_mask"] + [0] * pad)
            labels.append(x["labels"] + [-100] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long)
        }

rows = read_jsonl(DATA_PATH)
if len(rows) != 300:
    raise ValueError(f"Expected 300 SFT pairs, found {len(rows)}")

groups = {}
for row in rows:
    key = (row.get("difficulty", ""), row.get("answer_type", ""))
    groups.setdefault(key, []).append(row)

rng = random.Random(42)
train_rows = []
val_rows = []
for key, items in sorted(groups.items()):
    rng.shuffle(items)
    val_count = max(1, round(len(items) * 0.10))
    val_rows.extend(items[:val_count])
    train_rows.extend(items[val_count:])

rng.shuffle(train_rows)
rng.shuffle(val_rows)
train_ids = {x["id"] for x in train_rows}
val_ids = {x["id"] for x in val_rows}
assert not train_ids.intersection(val_ids)
assert len(train_rows) == 270 and len(val_rows) == 30

train_dataset = SFTDataset(train_rows)
val_dataset = SFTDataset(val_rows)

class EpochValidation(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        control.should_evaluate = True
        control.should_save = True
        return control
if train_dataset.supervised_answer_tokens <= 0:
    raise RuntimeError("Answer-only loss audit failed: no supervised answer tokens.")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
audit = {
    "records": len(train_dataset),
    "validation_records": len(val_dataset),
    "validation_masked_prompt_tokens": val_dataset.masked_prompt_tokens,
    "validation_supervised_answer_tokens": val_dataset.supervised_answer_tokens,
    "masked_prompt_tokens": train_dataset.masked_prompt_tokens,
    "supervised_answer_tokens": train_dataset.supervised_answer_tokens,
    "max_sequence_length": train_dataset.max_seen_length,
    "max_length_limit": MAX_LENGTH,
    "answer_only_loss": True,
    "held_out_test_set_loaded": False
}
(OUTPUT_DIR / "answer_only_audit.json").write_text(
    json.dumps(audit, indent=2), encoding="utf-8"
)

use_cuda = torch.cuda.is_available()
fp16 = bool(use_cuda and torch.cuda.get_device_capability(0)[0] < 8)
bf16 = bool(use_cuda and torch.cuda.is_bf16_supported())

training_args = TrainingArguments(**{k: v for k, v in dict(
    output_dir=str(OUTPUT_DIR),
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    optim="adamw_torch",
    logging_strategy="steps",
    logging_steps=5,
    save_strategy="epoch",
    save_total_limit=3,
    fp16=fp16 and not bf16,
    bf16=bf16,
    report_to="none",
    remove_unused_columns=False,
    dataloader_pin_memory=use_cuda,
    seed=42
).items() if k in inspect.signature(TrainingArguments.__init__).parameters})

trainer_kwargs = {
    "model": model,
    "args": training_args,
    "train_dataset": train_dataset,
    "eval_dataset": val_dataset,
    "callbacks": [EpochValidation()],
    "data_collator": AnswerOnlyCollator(tokenizer.pad_token_id)
}
if "processing_class" in inspect.signature(Trainer.__init__).parameters:
    trainer_kwargs["processing_class"] = tokenizer
else:
    trainer_kwargs["tokenizer"] = tokenizer

trainer = Trainer(**trainer_kwargs)
print("GPU available:", use_cuda)
if use_cuda:
    print("GPU:", torch.cuda.get_device_name(0))
print("Training records:", len(train_dataset))
print("Validation records:", len(val_dataset))
print("Answer-only audit:", audit)
print("Effective batch size:", BATCH_SIZE * GRAD_ACCUM)

result = trainer.train()
eval_logs = [x for x in trainer.state.log_history if "eval_loss" in x]
if len(eval_logs) < EPOCHS:
    raise RuntimeError(f"Expected validation loss for each epoch, found {len(eval_logs)}")
best_eval = min(eval_logs, key=lambda x: x["eval_loss"])
best_checkpoint = OUTPUT_DIR / f"checkpoint-{best_eval['step']}"
if not best_checkpoint.is_dir():
    raise FileNotFoundError(f"Best checkpoint not found: {best_checkpoint}")
final_model = AutoModelForCausalLM.from_pretrained(str(best_checkpoint))
final_model.config.use_cache = True
final_model.save_pretrained(str(OUTPUT_DIR / "sft_checkpoint"), safe_serialization=True)
tokenizer.save_pretrained(str(OUTPUT_DIR / "sft_checkpoint"))
(OUTPUT_DIR / "validation_results.json").write_text(
    json.dumps({"train_records": len(train_rows), "validation_records": len(val_rows), "per_epoch": eval_logs, "best_epoch": best_eval["epoch"], "best_eval_loss": best_eval["eval_loss"], "best_checkpoint": str(best_checkpoint)}, indent=2), encoding="utf-8"
)
(OUTPUT_DIR / "split_ids.json").write_text(
    json.dumps({"train_ids": sorted(train_ids), "validation_ids": sorted(val_ids), "seed": 42}, indent=2), encoding="utf-8"
)

(OUTPUT_DIR / "training_history.json").write_text(
    json.dumps(result.metrics, indent=2, default=str), encoding="utf-8"
)
run_config = {
    "base_checkpoint": CPT_CHECKPOINT,
    "data_path": str(DATA_PATH),
    "records": len(train_dataset),
    "validation_records": len(val_dataset),
    "validation_masked_prompt_tokens": val_dataset.masked_prompt_tokens,
    "validation_supervised_answer_tokens": val_dataset.supervised_answer_tokens,
    "epochs": EPOCHS,
    "per_device_train_batch_size": BATCH_SIZE,
    "gradient_accumulation_steps": GRAD_ACCUM,
    "effective_batch_size": BATCH_SIZE * GRAD_ACCUM,
    "learning_rate": LEARNING_RATE,
    "max_length": MAX_LENGTH,
    "special_tokens": SPECIAL_TOKENS,
    "answer_only_loss": True,
    "gpu_available": use_cuda,
    "gpu_name": torch.cuda.get_device_name(0) if use_cuda else None
}
(OUTPUT_DIR / "sft_run_config.json").write_text(
    json.dumps(run_config, indent=2), encoding="utf-8"
)

print("\nSFT TRAINING COMPLETED")
print("Final model:", OUTPUT_DIR / "sft_checkpoint")
print("Required file:", OUTPUT_DIR / "sft_checkpoint" / "model.safetensors")
print("Output folder:", OUTPUT_DIR)
print("Best validation loss:", best_eval["eval_loss"], "at epoch", best_eval["epoch"])
