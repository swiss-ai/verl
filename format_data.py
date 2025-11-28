import os
from datasets import load_dataset

# 1. Define your output paths exactly as requested
train_path = os.path.expanduser("~/scratch/data/gsm8k/train.parquet")
val_path = os.path.expanduser("~/scratch/data/gsm8k/test.parquet")

print("Downloading GSM8K...")
# Load the dataset (using 'main' configuration)
ds = load_dataset("gsm8k", "main")

# 2. Save Train Split
print(f"Saving train to {train_path}...")
ds['train'].to_parquet(train_path)

# 3. Save Test/Validation Split
print(f"Saving test to {val_path}...")
ds['test'].to_parquet(val_path)

print("Done!")