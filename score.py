"""Generate the three Phase 2 CSVs in a new directory."""
import argparse
import csv
import os
from pathlib import Path
from time import perf_counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("questions", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--sft-end-token", action="store_true")
    args = parser.parse_args()
    with args.questions.open(encoding="utf-8-sig", newline="") as handle:
        questions = list(csv.DictReader(handle))
    columns = ("id", "question", "expected_answer", "difficulty")
    if len(questions) != 30 or {q["id"] for q in questions} != {str(i) for i in range(1, 31)}:
        parser.error("Expected the 30 unique Phase 1 questions.")
    args.output.mkdir(parents=True, exist_ok=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = {
        "base": "openai-community/gpt2",
        "cpt": os.environ.get("CPT_CHECKPOINT") or "NeuraLabTeam/cpt-model",
        "sft": os.environ.get("SFT_CHECKPOINT") or "NeuraLabTeam/sft-model",
    }
    for name, checkpoint in models.items():
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device).eval()
        stop_ids = [tokenizer.eos_token_id]
        if name == "sft" and args.sft_end_token:
            stop_ids.append(tokenizer.get_vocab()["<|end|>"])
        fields = (*columns, "model_answer", "raw_answer", "score", "note", "latency")
        with (args.output / f"{name}_results.csv").open("x", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fields)
            writer.writeheader()
            for row in questions:
                question = row["question"]
                if name == "sft":
                    inputs = tokenizer.apply_chat_template(
                        [{"role": "user", "content": question}], add_generation_prompt=True,
                        tokenize=True, return_dict=True, return_tensors="pt",
                    )
                else:
                    inputs = tokenizer(f"Question: {question}\nAnswer:", return_tensors="pt")
                inputs = inputs.to(device)
                if device == "cuda":
                    torch.cuda.synchronize()
                start = perf_counter()
                with torch.inference_mode():
                    output = model.generate(
                        **inputs, do_sample=False, num_beams=1, max_new_tokens=100,
                        eos_token_id=stop_ids, pad_token_id=tokenizer.eos_token_id,
                    )
                if device == "cuda":
                    torch.cuda.synchronize()
                elapsed = perf_counter() - start
                tokens = output[0, inputs["input_ids"].shape[-1]:]
                result = {key: row[key] for key in columns}
                writer.writerow({
                    **result, "model_answer": tokenizer.decode(tokens, skip_special_tokens=True).strip(),
                    "raw_answer": tokenizer.decode(tokens), "latency": round(elapsed, 6),
                })
        print(f"{name}: {len(questions)} answers saved", flush=True)
        del model, tokenizer


if __name__ == "__main__":
    main()
