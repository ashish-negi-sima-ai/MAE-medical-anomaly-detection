#!/bin/bash

# Export BraTS MAE (pretrained) model to ONNX format
python export_onnx_mae.py \
  --model-path models/brats_pretrained.pth \
  --output-path exported_onnx_models/mae_brats_random.onnx \
  --dataset brats \
  --opset-version 14

python export_deterministic_mae.py \
  --model-path models/brats_pretrained.pth \
  --output-path exported_onnx_models/mae_brats_deterministic.onnx \
  --dataset brats \
  --opset-version 14