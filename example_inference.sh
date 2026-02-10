#!/bin/bash
#
# Example usage of simple_inference.py with TRUE POSITIVE examples
# Processes all correctly classified normal samples (True Negatives)
#

echo "============================================================"
echo "MAE Medical Anomaly Detection - Example Inference"
echo "Processing ALL correctly classified normal samples"
echo "============================================================"
echo ""

# Array of all True Negative examples (Normal → Correctly Predicted as Normal)
TRUE_NEGATIVES=(
    "brats_npy_data/test/normal/BraTS20_Training_355_slice_000.npy"
    "brats_npy_data/test/normal/BraTS20_Training_355_slice_001.npy"
    "brats_npy_data/test/normal/BraTS20_Training_355_slice_002.npy"
    "brats_npy_data/test/normal/BraTS20_Training_355_slice_003.npy"
    "brats_npy_data/test/normal/BraTS20_Training_355_slice_005.npy"
    "brats_npy_data/test/normal/BraTS20_Training_355_slice_007.npy"
    "brats_npy_data/test/normal/BraTS20_Training_355_slice_141.npy"
    "brats_npy_data/test/normal/BraTS20_Training_355_slice_143.npy"
    "brats_npy_data/test/normal/BraTS20_Training_355_slice_144.npy"
    "brats_npy_data/test/normal/BraTS20_Training_355_slice_145.npy"
    "brats_npy_data/test/normal/BraTS20_Training_355_slice_146.npy"
    "brats_npy_data/test/normal/BraTS20_Training_355_slice_147.npy"
    "brats_npy_data/test/normal/BraTS20_Training_355_slice_148.npy"
    "brats_npy_data/test/normal/BraTS20_Training_355_slice_149.npy"
    "brats_npy_data/test/normal/BraTS20_Training_355_slice_150.npy"
    "brats_npy_data/test/normal/BraTS20_Training_355_slice_151.npy"
    "brats_npy_data/test/normal/BraTS20_Training_355_slice_152.npy"
    "brats_npy_data/test/normal/BraTS20_Training_355_slice_153.npy"
    "brats_npy_data/test/normal/BraTS20_Training_355_slice_154.npy"
)

# True Positive Example: Abnormal brain slice correctly classified as ABNORMAL
ABNORMAL_NPY="brats_npy_data/test/abnormal/BraTS20_Training_001_slice_031.npy"

echo "Found ${#TRUE_NEGATIVES[@]} correctly classified normal samples"
echo ""

# Process all true negative examples
echo "Processing TRUE NEGATIVES (Normal → Predicted Normal) ✓"
echo "========================================================"
count=1
for npy_file in "${TRUE_NEGATIVES[@]}"; do
    if [ ! -f "$npy_file" ]; then
        echo "Warning: File not found: $npy_file"
        continue
    fi
    
    # Extract slice number from filename
    slice_num=$(basename "$npy_file" | sed 's/.*slice_\([0-9]*\)\.npy/\1/')
    
    echo "[$count/${#TRUE_NEGATIVES[@]}] Processing slice $slice_num..."
    
    python simple_inference.py \
        --input "$npy_file" \
        --mae-model models/brats_pretrained.pth \
        --classifier-model models/brats_finetuned.pth \
        --mask-ratio 0.75 \
        --num-trials 4 \
        --output "test_results/true_negative_slice_${slice_num}.png" \
        --no-viz > /dev/null 2>&1
    
    if [ $? -eq 0 ]; then
        echo "  ✓ Saved to: test_results/true_negative_slice_${slice_num}.png"
    else
        echo "  ✗ Failed to process"
    fi
    
    ((count++))
done

echo ""
echo "Processing TRUE POSITIVE (Abnormal → Predicted Abnormal) ✓"
echo "==========================================================="
if [ -f "$ABNORMAL_NPY" ]; then
    echo "Processing abnormal sample..."
    
    python simple_inference.py \
        --input "$ABNORMAL_NPY" \
        --mae-model models/brats_pretrained.pth \
        --classifier-model models/brats_finetuned.pth \
        --mask-ratio 0.75 \
        --num-trials 4 \
        --output test_results/true_positive_abnormal.png
    
    echo "  ✓ Saved to: test_results/true_positive_abnormal.png"
else
    echo "Error: Abnormal sample not found: $ABNORMAL_NPY"
fi

echo ""
echo "============================================================"
echo "Done! Processed ${#TRUE_NEGATIVES[@]} normal samples + 1 abnormal sample"
echo "Results saved in: test_results/"
echo "============================================================"


