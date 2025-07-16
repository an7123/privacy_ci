#!/bin/bash

# ==============================================================================
# == This script runs a series of experiments for different privacy defenses ==
# ==============================================================================

# --- Common Configuration ---
ARCH="resnet18"
SPLIT_LAYER="layer2"      # For ResNet-like models
CUTTING_LAYER=8           # For VGG-like models
DATASET="facescrub"         # Change to tiny_imagenet if you have it configured
EPOCHS=40           # A more reasonable number for larger datasets
BATCH_SIZE=64

# Learning Rates can differ per optimizer
LR_SGD=0.01               # Good starting point for SGD
LR_ADAM=1e-4              # Good starting point for Adam/AdamW
DATA_ROOT="./facescrub_processed_48x48"  

#

# Create checkpoint directory
mkdir -p checkpoints

# Function to print a separator
print_header() {
    echo ""
    echo "======================================================================"
    echo "  Running Experiment: $1"
    echo "======================================================================"
    echo ""
}


python3 main.py \
    --method sae \
    --arch $ARCH \
    --split-layer $SPLIT_LAYER \
    --cutting-layer $CUTTING_LAYER \
    --dataset $DATASET \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --lr $LR_ADAM \
    --top-p 0.03 \
    --hidden-scale 4 \
    --task-loss-weight 1.0 \
    --recon-loss-weight 0.3 \
    --noise-aware-training \
    --noise-sigma 0.5 \
    --client-ckpt "checkpoints/client_sae.pth" \
    --server-ckpt "checkpoints/server_sae.pth" \
    --sae-ckpt "checkpoints/sae_sae.pth" \
    --data-root $DATA_ROOT \
    --mask-epsilon 3.0 \
    --value-channel-budget 250 \
    --run-attack \


# # print_header "Randomized Top-K Baseline using AdamW"
# python3 main.py \
#     --method rand_topk \
#     --arch $ARCH \
#     --split-layer $SPLIT_LAYER \
#     --cutting-layer $CUTTING_LAYER \
#     --dataset $DATASET \
#     --epochs $EPOCHS \
#     --batch-size $BATCH_SIZE \
#     --lr $LR_ADAM \
#     --top-p 0.05 \
#     --hidden-scale 4 \
#     --task-loss-weight 1.0 \
#     --noise-aware-training \
#     --noise-sigma 1.0 \
#     --client-ckpt "checkpoints/client_rand_topk.pth" \
#     --server-ckpt "checkpoints/server_rand_topk.pth" \
#     --sae-ckpt "checkpoints/sae_rand_topk.pth" \
#     --data-root $DATA_ROOT \
#     --mask-epsilon 3.0 \
#     --value-channel-budget 300 \
#     #--use-nnc
#     # --run-attack \

echo ""
echo "======================================================================"
echo "  All experiments complete."
echo "======================================================================"