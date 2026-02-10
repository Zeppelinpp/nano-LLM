"""
Convert downloaded parquet file to training format (with tokenization)
Usage: uv run train/prepare_subset.py --data_dir ./data/openwebtext_subset
"""

import argparse
import os
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer


def prepare_subset(data_dir, tokenizer_name="gpt2", max_length=512):
    """Convert parquet file to tokenized dataset format"""

    parquet_path = os.path.join(data_dir, "train.parquet")
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(
            f"Parquet file not found at {parquet_path}. "
            f"Please download it first using:\n"
            f"  uv run train/download_subset.py --output_dir {data_dir}"
        )

    print(f"Loading parquet file from {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df)} examples")

    # Create Dataset from DataFrame
    dataset = Dataset.from_pandas(df)
    print(f"Created HuggingFace Dataset: {len(dataset)} examples")

    # Split into train and validation (90% train, 10% validation)
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = dataset["train"]
    val_dataset = dataset["test"]

    print(f"Split dataset: {len(train_dataset)} train, {len(val_dataset)} validation")

    # Load tokenizer
    print(f"\nLoading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token
    print(f"Vocab size: {tokenizer.vocab_size}")

    # Tokenize function
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding=False,
            max_length=max_length,
        )

    # Tokenize train dataset
    print(f"\nTokenizing train dataset ({len(train_dataset)} examples)...")
    train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        desc="Tokenizing train",
    )
    train_dataset = train_dataset.remove_columns(["text"])

    # Tokenize validation dataset
    print(f"Tokenizing validation dataset ({len(val_dataset)} examples)...")
    val_dataset = val_dataset.map(
        tokenize_function,
        batched=True,
        desc="Tokenizing validation",
    )
    val_dataset = val_dataset.remove_columns(["text"])

    print("✓ Tokenization completed!")

    # Save datasets to disk
    train_output_dir = os.path.join(data_dir, "train")
    val_output_dir = os.path.join(data_dir, "validation")

    os.makedirs(train_output_dir, exist_ok=True)
    os.makedirs(val_output_dir, exist_ok=True)

    train_dataset.save_to_disk(train_output_dir)
    val_dataset.save_to_disk(val_output_dir)

    print(f"\n✓ Dataset prepared successfully!")
    print(f"  Train data: {train_output_dir} ({len(train_dataset)} examples)")
    print(f"  Validation data: {val_output_dir} ({len(val_dataset)} examples)")
    print(f"\nYou can now train with:")
    print(f"  uv run train/pretrain.py --data_dir {data_dir} ...")


def main():
    parser = argparse.ArgumentParser(description="Prepare downloaded parquet file for training")
    parser.add_argument("--data_dir", type=str, required=True,
                       help="Directory containing train.parquet")
    parser.add_argument("--max_length", type=int, default=512,
                       help="Maximum sequence length for tokenization")
    args = parser.parse_args()

    prepare_subset(args.data_dir, max_length=args.max_length)


if __name__ == "__main__":
    main()
