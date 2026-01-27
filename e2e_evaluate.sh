#!/bin/bash

# Step 1: H5 → NPY
python h5_to_npy.py --h5-dir brats20_training_h5/ --output-base brats_npy_data/ --modality 1 --split test

# Step 2: Extract reconstructions (2-3 hours)
python extract_reconstructions.py --dataset brats --mask-ratio 0.75 --model-path models/brats_pretrained.pth --batch-size 64 --num-trials 4 --output-folder brats_reconstructions/test --test

# Step 3: Evaluate
python evaluate_sup.py --dataset brats --model-path models/brats_finetuned.pth --batch-size 64