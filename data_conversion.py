import pandas as pd
import os

# Define paths (Input and Output)
base_path = os.path.expanduser("~/scratch/data/gsm8k")
train_in = os.path.join(base_path, "train.parquet")
train_out = os.path.join(base_path, "train_chat.parquet")

val_base_path = os.path.expanduser("~/scratch/data/gsm8k")
val_in = os.path.join(val_base_path, "test.parquet")
val_out = os.path.join(val_base_path, "test_chat.parquet")

def convert_to_chat(row):
    # Create the standard chat structure
    return [
        {"role": "user", "content": row["question"]},
        {"role": "assistant", "content": row["answer"]}
    ]

def process_file(input_path, output_path):
    print(f"Processing {input_path}...")
    try:
        df = pd.read_parquet(input_path)
        
        # Apply the conversion
        df['messages'] = df.apply(convert_to_chat, axis=1)
        
        df_final = df[['messages']]
        df_final.to_parquet(output_path)
        print(f"Saved formatted data to {output_path}")
        print(f"Sample row:\n{df_final.iloc[0]['messages']}")
    except Exception as e:
        print(f"Error processing {input_path}: {e}")

# Process both splits
process_file(train_in, train_out)
process_file(val_in, val_out)