#!/bin/bash

# Export BraTS anomaly classifier to ONNX format
python export_onnx_classifier.py \
  --model-path models/brats_finetuned.pth \
  --output-path exported_onnx_models/classifier_brats.onnx \
  --dataset brats \
  --arch vit_base_patch16 \
  --opset-version 14 \
  --verify
