import argparse
import json
import os

import torch
import torch.nn as nn
from datasets import load_from_disk
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoTokenizer

from model.llms import GeneralLLM


class OpenWebTextDataLoader:
    """Data loader for OpenWebText dataset using existing tokenizer"""

    def __init__(self, tokenizer_name="gpt2", max_length=1024, batch_size=8):
        self.tokenizer_name = tokenizer_name
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.vocab_size = self.tokenizer.vocab_size

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_length = max_length
        self.batch_size = batch_size

        print(f"Using tokenizer: {tokenizer_name}")
        print(f"Vocab size: {self.vocab_size}")

    def load_from_local(self, data_dir, split="train"):
        """Load dataset from local disk"""
        split_dir = os.path.join(data_dir, split)

        if not os.path.exists(split_dir):
            raise FileNotFoundError(
                f"Dataset not found at {split_dir}. "
                f"Please download the dataset first using:\n"
                f"  uv run train/download_openwebtext.py --output_dir {data_dir}"
            )

        print(f"Loading {split} dataset from {split_dir}...")
        dataset = load_from_disk(split_dir)
        print(f"Loaded {len(dataset)} examples")
        return dataset

    def tokenize_dataset(self, dataset):
        """Tokenize dataset if not already tokenized"""
        # Check if dataset is already tokenized
        if "input_ids" in dataset.column_names:
            print("Dataset already tokenized, skipping tokenization")
            return dataset

        # Tokenize function
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding=False,
                max_length=self.max_length,
            )

        # Process dataset
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            desc="Tokenizing dataset",
        )

        # Remove text column if it exists
        if "text" in tokenized_dataset.column_names:
            tokenized_dataset = tokenized_dataset.remove_columns(["text"])

        return tokenized_dataset

    def collate_fn(self, examples):
        """Custom collation function for padding"""
        batch = self.tokenizer.pad(
            examples,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return batch

    def create_dataloader(self, dataset, shuffle=True):
        """Create PyTorch DataLoader"""
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            pin_memory=False,
            num_workers=0,
            collate_fn=self.collate_fn,
        )


def train_epoch(
    model,
    dataloader,
    optimizer,
    criterion,
    device,
    epoch,
    gradient_accumulation_steps=1,
    log_interval=100,
    writer=None,
):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0
    global_step = epoch * len(dataloader)

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    for batch_idx, batch in enumerate(pbar):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        # Forward pass
        logits, _ = model(input_ids[:, :-1], attention_mask=attention_mask[:, :-1])

        # Calculate loss
        loss = criterion(
            logits.reshape(-1, logits.size(-1)), input_ids[:, 1:].reshape(-1)
        )

        # Normalize loss for gradient accumulation
        loss = loss / gradient_accumulation_steps

        # Backward pass
        loss.backward()

        # Gradient accumulation
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        # Update metrics
        total_loss += loss.item() * gradient_accumulation_steps
        num_batches += 1

        pbar.set_postfix(
            {
                "loss": loss.item() * gradient_accumulation_steps,
                "avg_loss": total_loss / num_batches,
            }
        )

        # Log to TensorBoard at intervals
        if writer is not None and (batch_idx + 1) % log_interval == 0:
            step_loss = total_loss / num_batches
            current_step = global_step + batch_idx + 1
            writer.add_scalar("Train/Loss_step", step_loss, current_step)
            writer.add_scalar(
                "Train/Perplexity_step",
                torch.exp(torch.tensor(step_loss)),
                current_step,
            )
            writer.flush()  # Force write to disk

    avg_loss = total_loss / num_batches

    # Log epoch-level metrics to TensorBoard
    if writer is not None:
        writer.add_scalar("Train/Loss_epoch", avg_loss, epoch)
        writer.add_scalar(
            "Train/Perplexity_epoch", torch.exp(torch.tensor(avg_loss)), epoch
        )
        writer.flush()  # Force write to disk

    return avg_loss


def validate(model, dataloader, criterion, device, epoch, writer=None):
    """Validate model"""
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Validate]")
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            logits, _ = model(input_ids[:, :-1], attention_mask=attention_mask[:, :-1])
            loss = criterion(
                logits.reshape(-1, logits.size(-1)), input_ids[:, 1:].reshape(-1)
            )

            total_loss += loss.item()
            num_batches += 1

            pbar.set_postfix(
                {"loss": loss.item(), "avg_loss": total_loss / num_batches}
            )

    avg_loss = total_loss / num_batches

    # Log to TensorBoard
    if writer is not None:
        writer.add_scalar("Validation/Loss_epoch", avg_loss, epoch)
        writer.add_scalar(
            "Validation/Perplexity_epoch", torch.exp(torch.tensor(avg_loss)), epoch
        )
        writer.flush()  # Force write to disk

    return avg_loss


def main():
    parser = argparse.ArgumentParser(description="Train GeneralLLM on OpenWebText")

    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to local dataset directory (output of download_openwebtext.py)",
    )

    parser.add_argument("--vocab_size", type=int, default=50257)
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--max_position_embeddings", type=int, default=1024)

    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size (recommend 4-8 for 16GB MPS)",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Sequence length (recommend 512-1024 for 16GB MPS)",
    )
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Gradient accumulation for effective larger batch",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=100,
        help="Log metrics to TensorBoard every N batches",
    )

    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default="gpt2",
        help="Use existing tokenizer from HuggingFace (e.g., gpt2)",
    )

    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument(
        "--device", type=str, default="auto", help="auto (detect), mps, cuda, or cpu"
    )

    args = parser.parse_args()

    # Device detection
    if args.device == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            print(f"Using MPS device (Apple Silicon GPU)")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"Using CUDA device")
        else:
            device = torch.device("cpu")
            print(f"Using CPU")
    else:
        device = torch.device(args.device)
        print(f"Using device: {device}")

    # Initialize model
    model = GeneralLLM(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        max_position_embeddings=args.max_position_embeddings,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,} ({num_params / 1e6:.2f}M)")

    # Initialize dataset and dataloader
    print(f"\nLoading tokenizer: {args.tokenizer_name}")
    data_loader = OpenWebTextDataLoader(
        tokenizer_name=args.tokenizer_name,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )

    print(f"\nLoading data from {args.data_dir}...")
    train_dataset = data_loader.load_from_local(args.data_dir, split="train")
    val_dataset = data_loader.load_from_local(args.data_dir, split="validation")

    print("\nTokenizing datasets...")
    train_dataset = data_loader.tokenize_dataset(train_dataset)
    val_dataset = data_loader.tokenize_dataset(val_dataset)

    train_dataloader = data_loader.create_dataloader(train_dataset, shuffle=True)
    val_dataloader = data_loader.create_dataloader(val_dataset, shuffle=False)

    print(f"Train batches: {len(train_dataloader)}")
    print(f"Validation batches: {len(val_dataloader)}")

    # Optimizer and criterion
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=0.01
    )
    criterion = nn.CrossEntropyLoss()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize TensorBoard writer
    tensorboard_dir = os.path.join(args.output_dir, "tensorboard")
    os.makedirs(tensorboard_dir, exist_ok=True)
    writer = SummaryWriter(tensorboard_dir)
    print(f"TensorBoard logs will be saved to: {tensorboard_dir}")
    print(f"To view: tensorboard --logdir {tensorboard_dir}")

    # Save config
    config_path = os.path.join(args.output_dir, "config.json")
    with open(config_path, "w") as f:
        config = vars(args)
        config["total_params"] = num_params
        json.dump(config, f, indent=2)
    print(f"Saved config to: {config_path}\n")

    # Training loop
    best_val_loss = float("inf")
    train_losses = []
    val_losses = []

    print("Starting training...")
    print(f"Gradient accumulation: {args.gradient_accumulation_steps} steps\n")

    for epoch in range(args.num_epochs):
        # Train
        train_loss = train_epoch(
            model,
            train_dataloader,
            optimizer,
            criterion,
            device,
            epoch,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            log_interval=args.log_interval,
            writer=writer,
        )
        train_losses.append(train_loss)

        # Validate
        val_loss = validate(
            model, val_dataloader, criterion, device, epoch, writer=writer
        )
        val_losses.append(val_loss)

        print(
            f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}"
        )

        # Save checkpoint
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch}.pt")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(args.output_dir, "best_model.pt")
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved: {best_model_path}\n")
        else:
            print("")

    writer.close()

    print(f"\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
