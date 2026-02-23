#!/bin/bash

# Step 1: H5 → NPY
python convert_h5_to_npy.py

# Step 2: Extract reconstructions (2-3 hours)
# Reduced batch-size to 1 for 4GB VRAM (was 64)
python extract_reconstructions.py --dataset brats --mask-ratio 0.75 --model-path models/brats_pretrained.pth --batch-size 1 --num-trials 4 --output-folder brats_reconstructions/test --test

# Step 3: Evaluate
# Reduced batch-size to 1 for 4GB VRAM (was 64)
python evaluate_sup.py --dataset brats --model-path models/brats_finetuned.pth --batch-size 1 --output-folder brats_reconstructions/test