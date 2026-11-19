import os
import subprocess

folder = "/kaggle/input/datasets/toqa06ds/datascout-sft-validated-fix-zip"

os.environ["CPT_CHECKPOINT"] = "/kaggle/input/datasets/toqa06ds/datascout-ai-sft-300-training/checkpoint_epoch2/checkpoint_epoch2"
os.environ["SFT_DATA"] = folder + "/sft_pairs.jsonl"
os.environ["SFT_OUTPUT"] = "/kaggle/working/datascout_sft_300_validated"

subprocess.run(
    ["python", folder + "/train_sft.py"],
    env=os.environ.copy(),
    check=True
)
