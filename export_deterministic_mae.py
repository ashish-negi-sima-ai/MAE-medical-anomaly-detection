#!/usr/bin/env python3
"""
Export MAE model to ONNX with deterministic masking.

This script ensures that the ONNX model produces identical results to PyTorch
by baking in a fixed mask pattern (generated with a fixed seed).

For production use:
- If you need the SAME mask every time: Use this script
- If you need RANDOM masks: Use the regular export (randomness is expected)
"""

import warnings
import torch
import numpy as np
import argparse
import os

# Suppress TracerWarnings for cleaner output
warnings.filterwarnings('ignore', category=torch.jit.TracerWarning)

from export_onnx_mae import prepare_model, MAEForONNX


def export_deterministic(model_path, output_path, dataset='brats', opset_version=14):
    """Export MAE with deterministic behavior by fixing the random seed"""
    
    print("="*60)
    print("Exporting MAE with Deterministic Masking")
    print("="*60)
    
    # CRITICAL: Set seeds BEFORE creating the model wrapper
    # This ensures the mask pattern is deterministic
    print("\nSetting fixed random seeds for reproducibility...")
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Load model
    print(f"Loading model from: {model_path}")
    model, img_size = prepare_model(model_path, 'mae_vit_base_patch16')
    model.eval()
    
    # Wrap for ONNX with is_testing=True
    print("Wrapping model for ONNX export (is_testing=True)...")
    onnx_model = MAEForONNX(model, mask_ratio=0.75, is_testing=True)
    onnx_model.eval()
    
    # Create dummy input
    if dataset == 'brats':
        input_shape = (1, 1, 224, 224)
    else:
        input_shape = (1, 1, 64, 64)
    
    # CRITICAL: Set seeds again before creating dummy input
    torch.manual_seed(42)
    np.random.seed(42)
    dummy_input = torch.randn(input_shape)
    
    print(f"\nInput shape: {input_shape}")
    print(f"Output path: {output_path}")
    print(f"ONNX opset: {opset_version}")
    
    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # CRITICAL: Set seeds one more time before export
    # This ensures grid_masking uses the fixed seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Export to ONNX
    print("\nExporting to ONNX...")
    torch.onnx.export(
        onnx_model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input_image'],
        output_names=['reconstruction', 'mask'],
        dynamic_axes={
            'input_image': {0: 'batch_size'},
            'reconstruction': {0: 'batch_size'},
            'mask': {0: 'batch_size'}
        },
        verbose=False
    )
    
    print(f"✅ Successfully exported to: {output_path}")
    
    # Verify immediately
    print("\n" + "="*60)
    print("Verifying Export")
    print("="*60)
    
    import onnxruntime as ort
    
    # Prepare test input with SAME seed
    torch.manual_seed(42)
    np.random.seed(42)
    test_input_np = np.random.randn(*input_shape).astype(np.float32)
    test_input_torch = torch.from_numpy(test_input_np)
    
    # PyTorch inference
    print("\nRunning PyTorch inference...")
    torch.manual_seed(42)
    np.random.seed(42)
    
    with torch.no_grad():
        loss, pred, mask = model(test_input_torch, mask_ratio=0.75, is_testing=True)
        pytorch_recon = model.unpatchify(pred).cpu().numpy()
        pytorch_mask = mask.cpu().numpy()
    
    # ONNX inference  
    print("Running ONNX inference...")
    ort_session = ort.InferenceSession(output_path)
    onnx_outputs = ort_session.run(None, {ort_session.get_inputs()[0].name: test_input_np})
    onnx_recon = onnx_outputs[0]
    onnx_mask = onnx_outputs[1]
    
    # Compare
    mask_diff = np.abs(pytorch_mask - onnx_mask).max()
    recon_diff = np.abs(pytorch_recon - onnx_recon).max()
    
    print(f"\nResults:")
    print(f"  Mask max difference:          {mask_diff:.2e}")
    print(f"  Reconstruction max difference: {recon_diff:.2e}")
    
    if mask_diff < 1e-6 and recon_diff < 1e-3:
        print("\n✅ SUCCESS! PyTorch and ONNX outputs match perfectly!")
        return True
    else:
        print("\n⚠️  Outputs differ more than expected")
        if mask_diff > 1e-6:
            print(f"   Mask difference {mask_diff:.2e} indicates non-deterministic behavior")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Export MAE with deterministic masking')
    parser.add_argument('--model-path', type=str, default='models/brats_pretrained.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--output-path', type=str, 
                        default='exported_onnx_models/mae_brats_deterministic.onnx',
                        help='Output ONNX model path')
    parser.add_argument('--dataset', type=str, default='brats',
                        choices=['brats', 'luna16_unnorm'],
                        help='Dataset type (affects input size)')
    parser.add_argument('--opset-version', type=int, default=14,
                        help='ONNX opset version (use 14+ for attention ops)')
    
    args = parser.parse_args()
    
    success = export_deterministic(
        args.model_path,
        args.output_path,
        args.dataset,
        args.opset_version
    )
    
    exit(0 if success else 1)

