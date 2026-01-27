#!/usr/bin/env python3
"""
Export trained anomaly classifier (ViT) model to ONNX format.
This script exports the fine-tuned supervised classifier for anomaly detection.
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.onnx
import numpy as np

import models_vit


def safe_torch_load(checkpoint_path, device='cpu'):
    """
    Safely load PyTorch checkpoint with compatibility for different versions.
    """
    try:
        # Try loading normally first
        checkpoint = torch.load(checkpoint_path, map_location=device)
        print(f"✓ Checkpoint loaded successfully from {checkpoint_path}")
        return checkpoint
    except Exception as e:
        print(f"Standard loading failed: {e}")
        print("Trying with weights_only=False for PyTorch 2.6+ compatibility...")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            print(f"✓ Checkpoint loaded with weights_only=False")
            return checkpoint
        except Exception as e2:
            print(f"✗ Failed to load checkpoint: {e2}")
            raise


def prepare_model(chkpt_dir, arch='vit_large_patch16', num_classes=2, img_size=224, device='cpu'):
    """
    Load the trained ViT classifier model.
    
    Args:
        chkpt_dir: Path to checkpoint file
        arch: Model architecture name
        num_classes: Number of output classes (2 for binary anomaly detection)
        img_size: Input image size
        device: Device to load model on
    
    Returns:
        model: Loaded model in eval mode
    """
    print("\n" + "="*60)
    print("Loading ViT Classifier Model")
    print("="*60)
    print(f"Architecture: {arch}")
    print(f"Image size: {img_size}")
    print(f"Number of classes: {num_classes}")
    print(f"Device: {device}")
    print("="*60 + "\n")
    
    # Build model
    model = models_vit.__dict__[arch](
        num_classes=num_classes,
        drop_path_rate=0.0,
        global_pool=True,
        img_size=img_size
    )
    
    # Load checkpoint
    checkpoint = safe_torch_load(chkpt_dir, device)
    
    # Load state dict
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(f"Load state dict result: {msg}")
    
    # Move to device and set to eval mode
    model = model.to(device)
    model.eval()
    
    print("✓ Model loaded successfully\n")
    return model


class ViTClassifierForONNX(torch.nn.Module):
    """
    Wrapper for ViT classifier to make it ONNX-compatible.
    """
    def __init__(self, vit_model):
        super().__init__()
        self.vit_model = vit_model
    
    def forward(self, x):
        """
        Forward pass returns class logits.
        
        Args:
            x: Input tensor (B, C, H, W)
        
        Returns:
            logits: Class logits (B, num_classes)
        """
        logits = self.vit_model(x)
        return logits


def export_classifier_to_onnx(
    model_path,
    output_path,
    img_size=224,
    batch_size=1,
    opset_version=14,
    device='cpu',
    arch='vit_large_patch16',
    num_classes=2
):
    """
    Export ViT classifier to ONNX format.
    
    Args:
        model_path: Path to trained model checkpoint
        output_path: Path to save ONNX model
        img_size: Input image size (default: 224 for BraTS)
        batch_size: Batch size for export (use 'dynamic' for dynamic batching)
        opset_version: ONNX opset version
        device: Device to use
        arch: Model architecture
        num_classes: Number of output classes
    """
    print("\n" + "="*60)
    print("ONNX Export - ViT Anomaly Classifier")
    print("="*60)
    
    # Load model
    model = prepare_model(model_path, arch=arch, num_classes=num_classes, 
                         img_size=img_size, device=device)
    
    # Wrap model for ONNX
    onnx_model = ViTClassifierForONNX(model)
    onnx_model.eval()
    
    # Create dummy input (single channel for medical images)
    dummy_input = torch.randn(1 if batch_size == 'dynamic' else batch_size, 
                              1, img_size, img_size, device=device)
    
    print(f"Dummy input shape: {dummy_input.shape}")
    
    # Test forward pass
    print("\nTesting forward pass...")
    with torch.no_grad():
        output = onnx_model(dummy_input)
        print(f"✓ Forward pass successful")
        print(f"  Output shape: {output.shape}")
        print(f"  Output (logits): {output}")
    
    # Define dynamic axes if needed
    dynamic_axes = None
    if batch_size == 'dynamic':
        dynamic_axes = {
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
        print(f"\n✓ Using dynamic batch size")
    
    # Export to ONNX
    print(f"\nExporting to ONNX (opset {opset_version})...")
    print(f"Output path: {output_path}")
    
    # Create output directory
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    try:
        torch.onnx.export(
            onnx_model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=dynamic_axes,
            verbose=False
        )
        
        print(f"\n{'='*60}")
        print("✓ Export successful!")
        print(f"{'='*60}")
        print(f"Model saved to: {output_path}")
        print(f"Input shape: (batch_size, 1, {img_size}, {img_size})")
        print(f"Output shape: (batch_size, {num_classes})")
        print(f"{'='*60}\n")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error during export: {e}")
        print("\nTroubleshooting tips:")
        print("1. Try with --opset-version 14 or higher")
        print("2. Check that model loads correctly (--test-loading)")
        print("3. Ensure sufficient memory")
        print("4. Verify PyTorch and ONNX versions are compatible")
        return False


def verify_onnx_model(onnx_path, model_path, img_size=224, arch='vit_large_patch16', 
                      num_classes=2, device='cpu'):
    """
    Verify ONNX model by comparing outputs with PyTorch model.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        print("✗ onnxruntime not installed. Install with: pip install onnxruntime")
        return False
    
    print("\n" + "="*60)
    print("Verifying ONNX Model")
    print("="*60)
    
    # Load PyTorch model
    pytorch_model = prepare_model(model_path, arch=arch, num_classes=num_classes,
                                  img_size=img_size, device=device)
    pytorch_model.eval()
    
    # Load ONNX model
    print(f"\nLoading ONNX model from: {onnx_path}")
    ort_session = ort.InferenceSession(onnx_path)
    
    # Create test input
    test_input = torch.randn(1, 1, img_size, img_size, device=device)
    
    # PyTorch inference
    with torch.no_grad():
        pytorch_output = pytorch_model(test_input)
    
    # ONNX inference
    onnx_input = {ort_session.get_inputs()[0].name: test_input.cpu().numpy()}
    onnx_output = ort_session.run(None, onnx_input)[0]
    
    # Compare outputs
    pytorch_output_np = pytorch_output.cpu().numpy()
    max_diff = np.abs(pytorch_output_np - onnx_output).max()
    
    print(f"\nPyTorch output: {pytorch_output_np}")
    print(f"ONNX output:    {onnx_output}")
    print(f"Max difference: {max_diff}")
    
    if max_diff < 1e-4:
        print("\n✓ Verification successful! Outputs match.")
        return True
    else:
        print(f"\n⚠ Warning: Outputs differ by {max_diff}")
        print("This might be acceptable depending on your use case.")
        return True


def main():
    parser = argparse.ArgumentParser(
        description='Export ViT anomaly classifier to ONNX format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export BraTS classifier with verification
  python export_onnx_classifier.py \\
      --model-path models/brats_finetuned.pth \\
      --output-path exports/classifier_brats.onnx \\
      --dataset brats \\
      --verify

  # Export with dynamic batch size
  python export_onnx_classifier.py \\
      --model-path models/brats_finetuned.pth \\
      --output-path exports/classifier_brats_dynamic.onnx \\
      --dataset brats \\
      --batch-size dynamic

  # Test loading only
  python export_onnx_classifier.py \\
      --model-path models/brats_finetuned.pth \\
      --dataset brats \\
      --test-loading
        """
    )
    
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained classifier checkpoint (.pth)')
    parser.add_argument('--output-path', type=str, default='exports/classifier.onnx',
                       help='Output path for ONNX model')
    parser.add_argument('--dataset', type=str, default='brats',
                       choices=['brats', 'luna16'],
                       help='Dataset name (determines img_size)')
    parser.add_argument('--arch', type=str, default='vit_large_patch16',
                       choices=['vit_base_patch16', 'vit_large_patch16', 'vit_huge_patch14'],
                       help='Model architecture')
    parser.add_argument('--num-classes', type=int, default=2,
                       help='Number of output classes (default: 2 for binary)')
    parser.add_argument('--batch-size', type=str, default='1',
                       help='Batch size (use "dynamic" for dynamic batching)')
    parser.add_argument('--opset-version', type=int, default=14,
                       help='ONNX opset version (default: 14, min: 14 for attention)')
    parser.add_argument('--verify', action='store_true',
                       help='Verify ONNX model after export')
    parser.add_argument('--test-loading', action='store_true',
                       help='Only test model loading without export')
    parser.add_argument('--check-deps', action='store_true',
                       help='Check if all dependencies are installed')
    
    args = parser.parse_args()
    
    # Check dependencies
    if args.check_deps:
        print("Checking dependencies...")
        print(f"✓ PyTorch: {torch.__version__}")
        try:
            import onnx
            print(f"✓ ONNX: {onnx.__version__}")
        except ImportError:
            print("✗ ONNX not installed (pip install onnx)")
        try:
            import onnxruntime
            print(f"✓ ONNX Runtime: {onnxruntime.__version__}")
        except ImportError:
            print("✗ ONNX Runtime not installed (pip install onnxruntime)")
        print("\n✓ Dependency check complete")
        return
    
    # Determine image size based on dataset
    img_size = 224 if args.dataset == 'brats' else 64
    
    # Determine batch size
    if args.batch_size.lower() == 'dynamic':
        batch_size = 'dynamic'
    else:
        try:
            batch_size = int(args.batch_size)
        except ValueError:
            print(f"✗ Invalid batch size: {args.batch_size}")
            print("Use an integer or 'dynamic'")
            sys.exit(1)
    
    # Detect device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Test loading only
    if args.test_loading:
        print("\nTesting model loading only...")
        try:
            model = prepare_model(args.model_path, arch=args.arch, 
                                num_classes=args.num_classes, 
                                img_size=img_size, device=device)
            print("✓ Model loaded successfully!")
            
            # Test forward pass
            dummy_input = torch.randn(1, 1, img_size, img_size, device=device)
            with torch.no_grad():
                output = model(dummy_input)
            print(f"✓ Forward pass successful! Output shape: {output.shape}")
            
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            sys.exit(1)
        return
    
    # Export to ONNX
    success = export_classifier_to_onnx(
        model_path=args.model_path,
        output_path=args.output_path,
        img_size=img_size,
        batch_size=batch_size,
        opset_version=args.opset_version,
        device=device,
        arch=args.arch,
        num_classes=args.num_classes
    )
    
    if not success:
        sys.exit(1)
    
    # Verify if requested
    if args.verify:
        verify_success = verify_onnx_model(
            args.output_path,
            args.model_path,
            img_size=img_size,
            arch=args.arch,
            num_classes=args.num_classes,
            device=device
        )
        if not verify_success:
            sys.exit(1)
    
    print("\n✓ All done!")


if __name__ == '__main__':
    main()

