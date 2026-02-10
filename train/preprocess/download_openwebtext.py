"""
Download OpenWebText dataset to local directory
Usage: uv run train/download_openwebtext.py --output_dir ./data/openwebtext
"""

import argparse
from datasets import load_dataset
import os


def download_openwebtext(output_dir, num_examples=None):
    """Download OpenWebText dataset to local directory"""

    print(f"Downloading OpenWebText dataset to {output_dir}...")
    print("This may take a while depending on your internet connection.\n")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load dataset (non-streaming mode to download everything)
    print("Loading OpenWebText dataset...")
    dataset = load_dataset("openwebtext")

    print(f"Dataset loaded successfully!")
    print(f"Train split: {len(dataset['train'])} examples\n")

    # Save dataset to disk
    print("Saving dataset to local disk...")

    # Save train split
    train_output_dir = os.path.join(output_dir, "train")
    os.makedirs(train_output_dir, exist_ok=True)

    if num_examples:
        # Save subset
        print(f"Saving {num_examples} examples from train split...")
        subset = dataset["train"].select(
            range(min(num_examples, len(dataset["train"])))
        )
        subset.save_to_disk(train_output_dir)
        print(f"Saved {len(subset)} examples to {train_output_dir}")
    else:
        # Save full dataset
        print(f"Saving all {len(dataset['train'])} examples from train split...")
        dataset["train"].save_to_disk(train_output_dir)
        print(f"Saved {len(dataset['train'])} examples to {train_output_dir}")

    # Create a small validation split (if not too large)
    val_size = min(
        10000, len(dataset["train"]) // 10
    )  # 10% or 10k, whichever is smaller
    print(f"\nCreating validation split with {val_size} examples...")
    val_dataset = dataset["train"].select(
        range(len(dataset["train"]) - val_size, len(dataset["train"]))
    )

    val_output_dir = os.path.join(output_dir, "validation")
    os.makedirs(val_output_dir, exist_ok=True)
    val_dataset.save_to_disk(val_output_dir)
    print(f"Saved {len(val_dataset)} validation examples to {val_output_dir}")

    # Save info file
    info_path = os.path.join(output_dir, "dataset_info.txt")
    with open(info_path, "w") as f:
        f.write(f"OpenWebText Dataset\n")
        f.write(f"===================\n\n")
        f.write(f"Downloaded: {num_examples if num_examples else 'Full dataset'}\n")
        f.write(
            f"Train examples: {len(subset) if num_examples else len(dataset['train'])}\n"
        )
        f.write(f"Validation examples: {len(val_dataset)}\n")
        f.write(
            f"Total examples: {len(subset) if num_examples else len(dataset['train'])} + {len(val_dataset)}\n"
        )

    print(f"\n✓ Dataset downloaded successfully!")
    print(f"  Train data: {train_output_dir}")
    print(f"  Validation data: {val_output_dir}")
    print(f"  Info file: {info_path}")
    print(f"\nYou can now train the model using:")
    print(f"  uv run train/pretrain.py --data_dir {output_dir} ...")


def main():
    parser = argparse.ArgumentParser(description="Download OpenWebText dataset")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/openwebtext",
        help="Output directory for the dataset",
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=None,
        help="Number of examples to download (None for full dataset)",
    )
    args = parser.parse_args()

    download_openwebtext(args.output_dir, args.num_examples)


if __name__ == "__main__":
    main()
