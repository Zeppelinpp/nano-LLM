#!/bin/bash

# Activate virtual environment
source .venv/bin/activate

# === Training Configuration ===
BATCH_SIZE=4
LR=6e-4
MAX_LENGTH=1024
NUM_LAYERS=12
NUM_HEADS=12
MAX_POSITION_EMBEDDINGS=1024
GRADIENT_ACCUMULATION_STEPS=4
LOG_INTERVAL=20
NUM_EPOCHS=3
DATA_DIR=./data/openwebtext_subset
OUTPUT_DIR=./checkpoints_test

# Display training configuration
echo "=========================================="
echo "Training Configuration"
echo "=========================================="
echo "Batch Size: $BATCH_SIZE"
echo "Learning Rate: $LR"
echo "Max Length: $MAX_LENGTH"
echo "Num Layers: $NUM_LAYERS"
echo "Num Heads: $NUM_HEADS"
echo "Max Position Embeddings: $MAX_POSITION_EMBEDDINGS"
echo "Gradient Accumulation Steps: $GRADIENT_ACCUMULATION_STEPS"
echo "Log Interval: $LOG_INTERVAL (batches)"
echo "Num Epochs: $NUM_EPOCHS"
echo "Data Directory: $DATA_DIR"
echo "Output Directory: $OUTPUT_DIR"
echo "=========================================="
echo ""

# Run training
python train/pretrain.py \
  --data_dir $DATA_DIR \
  --batch_size $BATCH_SIZE \
  --learning_rate $LR \
  --max_length $MAX_LENGTH \
  --num_layers $NUM_LAYERS \
  --num_heads $NUM_HEADS \
  --max_position_embeddings $MAX_POSITION_EMBEDDINGS \
  --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
  --log_interval $LOG_INTERVAL \
  --num_epochs $NUM_EPOCHS \
  --output_dir $OUTPUT_DIR \
  --vocab_size 50257

echo ""
echo "Training completed!"
echo "Checkpoints saved to: $OUTPUT_DIR"
echo "View TensorBoard: tensorboard --logdir $OUTPUT_DIR/tensorboard"
