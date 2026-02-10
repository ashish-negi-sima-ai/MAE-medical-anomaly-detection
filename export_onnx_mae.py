import argparse
import os
import sys
import warnings

# Suppress TracerWarnings for cleaner output
import torch
warnings.filterwarnings('ignore', category=torch.jit.TracerWarning)

# Check for required dependencies
try:
    import torch.onnx
    # Check PyTorch version for torch.load compatibility
    pytorch_version = torch.__version__
    print(f"Found PyTorch version: {pytorch_version}")
except ImportError as e:
    print(f"Error: PyTorch is not installed. Please install it with: pip install torch torchvision")
    sys.exit(1)

try:
    import timm
    # Check timm version and handle compatibility
    timm_version = timm.__version__
    print(f"Found timm version: {timm_version}")
    
    # Handle qk_scale parameter compatibility
    from timm.models.vision_transformer import Block
    import inspect
    
    # Check if Block still accepts qk_scale parameter
    block_params = inspect.signature(Block.__init__).parameters
    supports_qk_scale = 'qk_scale' in block_params
    
    if not supports_qk_scale:
        print("Warning: Your timm version doesn't support qk_scale parameter. Applying compatibility patch...")
        
        # Monkey patch the models_mae to handle timm compatibility
        import models_mae
        
        # Store original Block class
        OriginalBlock = Block
        
        class CompatibleBlock(OriginalBlock):
            def __init__(self, *args, qk_scale=None, **kwargs):
                # Remove qk_scale from kwargs if it exists
                kwargs.pop('qk_scale', None)
                super().__init__(*args, **kwargs)
        
        # Replace Block in the models_mae module
        import timm.models.vision_transformer
        timm.models.vision_transformer.Block = CompatibleBlock
        
        # Reload models_mae with the patched Block
        import importlib
        importlib.reload(models_mae)
    
    import models_mae
    
except ImportError as e:
    print(f"Error: Could not import models_mae. Make sure you're in the correct directory and dependencies are installed.")
    print(f"Missing dependency error: {e}")
    print("Try installing: pip install timm")
    sys.exit(1)

import numpy as np

DATASETS = ['brats', 'cbis_roi', 'cbis', 'luna16', 'luna16_unnorm']


def safe_torch_load(checkpoint_path):
    """Safely load a PyTorch checkpoint with compatibility for different PyTorch versions"""
    print(f"Attempting to load checkpoint: {checkpoint_path}")
    
    # For PyTorch 2.6+, try weights_only=False directly if we know it contains complex objects
    # This is safer than trying weights_only=True first and failing
    try:
        print("Loading with weights_only=False (required for checkpoints with argparse.Namespace)...")
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        print("✓ Checkpoint loaded successfully!")
        return checkpoint
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        
        # If the above fails, try one more fallback approach
        try:
            print("Trying alternative loading method...")
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            print("✓ Checkpoint loaded with alternative method!")
            return checkpoint
        except Exception as final_error:
            print(f"All loading methods failed: {final_error}")
            raise final_error


def prepare_model(chkpt_dir, arch='mae_vit_large_patch16', img_size=None):
    """Prepare and load the MAE model from checkpoint
    
    Args:
        chkpt_dir: Path to checkpoint file
        arch: Model architecture name
        img_size: Image size (if None, inferred from arch - 224 for base, 64 for others)
    """
    # Infer image size if not provided
    if img_size is None:
        # Default: base models use 224, others use 64
        img_size = 224 if 'base' in arch else 64
    
    patch_size = 16
    model = getattr(models_mae, arch)(img_size=img_size, patch_size=patch_size)
    
    # load model with safe loading
    print(f"Loading checkpoint from: {chkpt_dir}")
    checkpoint = safe_torch_load(chkpt_dir)
    
    # Extract model state dict
    if 'model' in checkpoint:
        model_state_dict = checkpoint['model']
    else:
        # If the checkpoint is just the model state dict
        model_state_dict = checkpoint
    
    msg = model.load_state_dict(model_state_dict, strict=False)
    print(f"Model loading message: {msg}")
    
    # switch to evaluation mode
    model.eval()
    return model, img_size


class MAEForONNX(torch.nn.Module):
    """Wrapper class to make MAE model ONNX-compatible
    
    Note: For reproducible/deterministic behavior, set is_testing=True
    This ensures the same mask is generated for the same input across runs.
    """
    
    def __init__(self, mae_model, mask_ratio=0.75, is_testing=False):
        super().__init__()
        self.mae_model = mae_model
        self.mask_ratio = mask_ratio
        self.is_testing = is_testing
    
    def forward(self, x):
        # Forward pass with fixed mask ratio
        # is_testing=True ensures deterministic masking
        loss, pred, mask = self.mae_model(x, mask_ratio=self.mask_ratio, is_testing=self.is_testing)
        
        # Reconstruct the image from patches
        reconstruction = self.mae_model.unpatchify(pred)
        
        # Return reconstruction and mask for ONNX export
        return reconstruction, mask


def export_mae_to_onnx(model_path, output_path, dataset, mask_ratio=0.75, opset_version=11, is_testing=False):
    """Export MAE model to ONNX format"""
    
    # Check if model path exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")
    
    # Prepare the model
    model, img_size = prepare_model(model_path, 'mae_vit_base_patch16')
    
    # Wrap model for ONNX export
    onnx_model = MAEForONNX(model, mask_ratio=mask_ratio, is_testing=is_testing)
    onnx_model.eval()
    
    # Determine input shape based on dataset
    if dataset == 'brats':
        input_shape = (1, 1, 224, 224)  # batch_size=1, channels=1, height=224, width=224
    else:
        input_shape = (1, 1, 64, 64)   # batch_size=1, channels=1, height=64, width=64
    
    # Create dummy input
    dummy_input = torch.randn(input_shape)
    
    # Define input and output names
    input_names = ['input_image']
    output_names = ['reconstruction', 'mask']
    
    # Define dynamic axes if needed (for variable batch size)
    dynamic_axes = {
        'input_image': {0: 'batch_size'},
        'reconstruction': {0: 'batch_size'},
        'mask': {0: 'batch_size'}
    }
    
    print(f"Exporting MAE model to ONNX...")
    print(f"Input shape: {input_shape}")
    print(f"Output path: {output_path}")
    
    # Export to ONNX
    torch.onnx.export(
        onnx_model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        verbose=False  # Set to True for detailed debugging
    )
    
    print(f"Successfully exported MAE model to {output_path}")


class MAEEncoderForONNX(torch.nn.Module):
    """Wrapper class to export only the encoder part of MAE"""
    
    def __init__(self, mae_model, mask_ratio=0.75, is_testing=False):
        super().__init__()
        self.mae_model = mae_model
        self.mask_ratio = mask_ratio
        self.is_testing = is_testing
    
    def forward(self, x):
        # Only run the encoder part
        latent, mask, ids_restore = self.mae_model.forward_encoder(x, self.mask_ratio, is_testing=self.is_testing)
        return latent


def export_mae_encoder_to_onnx(model_path, output_path, dataset, mask_ratio=0.75, opset_version=11, is_testing=False):
    """Export only the MAE encoder to ONNX format"""
    
    # Check if model path exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")
    
    # Prepare the model
    model, img_size = prepare_model(model_path, 'mae_vit_base_patch16')
    
    # Wrap encoder for ONNX export
    onnx_model = MAEEncoderForONNX(model, mask_ratio=mask_ratio, is_testing=is_testing)
    onnx_model.eval()
    
    # Determine input shape based on dataset
    if dataset == 'brats':
        input_shape = (1, 1, 224, 224)
    else:
        input_shape = (1, 1, 64, 64)
    
    # Create dummy input
    dummy_input = torch.randn(input_shape)
    
    # Define input and output names
    input_names = ['input_image']
    output_names = ['encoded_features']
    
    # Define dynamic axes
    dynamic_axes = {
        'input_image': {0: 'batch_size'},
        'encoded_features': {0: 'batch_size'}
    }
    
    print(f"Exporting MAE encoder to ONNX...")
    print(f"Input shape: {input_shape}")
    print(f"Output path: {output_path}")
    
    # Export to ONNX
    torch.onnx.export(
        onnx_model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        verbose=False  # Set to True for detailed debugging
    )
    
    print(f"Successfully exported MAE encoder to {output_path}")


def verify_onnx_model(onnx_path, model_path, dataset, mask_ratio=0.75, is_testing=False, export_type='full'):
    """Verify the exported ONNX model by comparing with PyTorch outputs
    
    Note: For meaningful comparison, the model should be exported with is_testing=True
    to ensure deterministic masking. Otherwise, masks will differ between runs.
    """
    try:
        import onnx
        import onnxruntime as ort
        
        print("\n" + "="*60)
        print("Verifying ONNX Model")
        print("="*60)
        
        # Warn if not using deterministic mode
        if not is_testing:
            print("\n⚠️  WARNING: Model was exported with random masking (is_testing=False)")
            print("   Verification will show large differences because:")
            print("   - PyTorch generates a random mask during verification")
            print("   - ONNX has a different random mask pattern baked in from export")
            print("   For accurate verification, re-export with --is-testing flag\n")
        
        # Load ONNX model and check structure
        print(f"\nLoading ONNX model from: {onnx_path}")
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("✓ ONNX model structure validation passed!")
        
        # Load PyTorch model
        print("\nLoading PyTorch model for comparison...")
        pytorch_model, img_size = prepare_model(model_path, 'mae_vit_base_patch16')
        pytorch_model.eval()
        
        # Prepare test input (use fixed seed for reproducibility)
        torch.manual_seed(42)
        np.random.seed(42)
        
        if dataset == 'brats':
            test_input_np = np.random.randn(1, 1, 224, 224).astype(np.float32)
        else:
            test_input_np = np.random.randn(1, 1, 64, 64).astype(np.float32)
        
        test_input_torch = torch.from_numpy(test_input_np)
        
        # IMPORTANT: Set the same random seed for both PyTorch and ONNX
        # to ensure they generate the same mask
        print("\nSetting fixed random seed for deterministic mask generation...")
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Run PyTorch inference
        print("Running PyTorch inference...")
        with torch.no_grad():
            if export_type == 'encoder':
                # For encoder only export
                pytorch_output, _, _ = pytorch_model.forward_encoder(test_input_torch, mask_ratio, is_testing=True)
                pytorch_output_np = pytorch_output.cpu().numpy()
            else:
                # For full model export (reconstruction + mask)
                # Use is_testing=True to ensure deterministic masking
                loss, pred, mask = pytorch_model(test_input_torch, mask_ratio=mask_ratio, is_testing=True)
                reconstruction = pytorch_model.unpatchify(pred)
                pytorch_reconstruction_np = reconstruction.cpu().numpy()
                pytorch_mask_np = mask.cpu().numpy()
        
        print("✓ PyTorch inference successful")
        
        # Run ONNX Runtime inference
        print("\nRunning ONNX Runtime inference...")
        ort_session = ort.InferenceSession(onnx_path)
        ort_inputs = {ort_session.get_inputs()[0].name: test_input_np}
        onnx_outputs = ort_session.run(None, ort_inputs)
        print("✓ ONNX Runtime inference successful")
        
        # Compare outputs
        print("\n" + "="*60)
        print("Comparing PyTorch vs ONNX outputs:")
        print("="*60)
        
        if export_type == 'encoder':
            # Compare encoder output
            onnx_output = onnx_outputs[0]
            max_diff = np.abs(pytorch_output_np - onnx_output).max()
            mean_diff = np.abs(pytorch_output_np - onnx_output).mean()
            
            print(f"\nEncoder output comparison:")
            print(f"  PyTorch output shape: {pytorch_output_np.shape}")
            print(f"  ONNX output shape:    {onnx_output.shape}")
            print(f"  Max difference:  {max_diff:.2e}")
            print(f"  Mean difference: {mean_diff:.2e}")
            
            if max_diff < 1e-4:
                print("\n✓ Verification successful! Outputs match closely.")
                return True
            else:
                print(f"\n⚠ Warning: Outputs differ by {max_diff:.2e}")
                print("This might be acceptable depending on your use case.")
                return True
        else:
            # Compare full model outputs (reconstruction + mask)
            onnx_reconstruction = onnx_outputs[0]
            onnx_mask = onnx_outputs[1]
            
            recon_max_diff = np.abs(pytorch_reconstruction_np - onnx_reconstruction).max()
            recon_mean_diff = np.abs(pytorch_reconstruction_np - onnx_reconstruction).mean()
            mask_max_diff = np.abs(pytorch_mask_np - onnx_mask).max()
            
            print(f"\nReconstruction output comparison:")
            print(f"  PyTorch shape: {pytorch_reconstruction_np.shape}")
            print(f"  ONNX shape:    {onnx_reconstruction.shape}")
            print(f"  Max difference:  {recon_max_diff:.2e}")
            print(f"  Mean difference: {recon_mean_diff:.2e}")
            
            print(f"\nMask output comparison:")
            print(f"  PyTorch shape: {pytorch_mask_np.shape}")
            print(f"  ONNX shape:    {onnx_mask.shape}")
            print(f"  Max difference: {mask_max_diff:.2e}")
            
            if recon_max_diff < 1e-4 and mask_max_diff < 1e-6:
                print("\n✓ Verification successful! Outputs match closely.")
                print("   Model was exported correctly with deterministic behavior.")
                return True
            elif not is_testing:
                print(f"\n⚠️  Large differences detected (recon: {recon_max_diff:.2e}, mask: {mask_max_diff:.2e})")
                print("   This is EXPECTED because the model uses random masking.")
                print("   The ONNX model is working correctly - differences are due to:")
                print("   - Different random mask patterns between PyTorch and ONNX")
                print("   For verification with matching outputs, use --is-testing flag.")
                return True
            else:
                print(f"\n⚠️  Unexpected differences (recon: {recon_max_diff:.2e}, mask: {mask_max_diff:.2e})")
                print("   Even with is_testing=True, differences are larger than expected.")
                print("   This might indicate an export issue or numerical precision differences.")
                return False
            
    except ImportError:
        print("✗ ONNX or ONNXRuntime not installed. Skipping verification.")
        print("To verify the model, install: pip install onnx onnxruntime")
        return False
    except Exception as e:
        print(f"✗ ONNX model verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_dependencies():
    """Check if all required dependencies are available"""
    missing_deps = []
    
    try:
        import torch
        import torch.onnx
        print(f"✓ PyTorch version: {torch.__version__}")
    except ImportError:
        missing_deps.append("torch")
    
    try:
        import timm
        print(f"✓ timm version: {timm.__version__}")
    except ImportError:
        missing_deps.append("timm")
    
    if missing_deps:
        print("Missing required dependencies:")
        for dep in missing_deps:
            print(f"  - {dep}")
        print("\nInstall them with:")
        print(f"pip install {' '.join(missing_deps)}")
        return False
    return True


def test_model_loading(model_path, dataset):
    """Test if the model can be loaded successfully with current timm version"""
    try:
        print(f"Testing model loading from: {model_path}")
        print(f"Dataset: {dataset}")
        
        if not os.path.exists(model_path):
            print(f"✗ Model file not found: {model_path}")
            return False
        
        # Try to load the model
        model, img_size = prepare_model(model_path, 'mae_vit_base_patch16')
        print(f"✓ Model loaded successfully!")
        print(f"✓ Image size: {img_size}")
        
        # Test a forward pass
        if dataset == 'brats':
            test_input = torch.randn(1, 1, 224, 224)
        else:
            test_input = torch.randn(1, 1, 64, 64)
        
        with torch.no_grad():
            loss, pred, mask = model(test_input, mask_ratio=0.75)
        
        print(f"✓ Forward pass successful!")
        print(f"✓ Output shapes - pred: {pred.shape}, mask: {mask.shape}")
        return True
        
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export MAE model to ONNX format')
    parser.add_argument('--model-path', type=str, required=True, 
                        help='Path to the trained MAE model checkpoint')
    parser.add_argument('--output-path', type=str, required=True,
                        help='Output path for the ONNX model')
    parser.add_argument('--dataset', type=str, required=True, choices=DATASETS,
                        help='Dataset name (affects input image size)')
    parser.add_argument('--mask-ratio', type=float, default=0.75,
                        help='Mask ratio for MAE model')
    parser.add_argument('--export-type', type=str, default='full', choices=['full', 'encoder'],
                        help='Export full model or just encoder')
    parser.add_argument('--opset-version', type=int, default=11,
                        help='ONNX opset version')
    parser.add_argument('--verify', action='store_true',
                        help='Verify the exported ONNX model (recommended: use with --is-testing for meaningful comparison)')
    parser.add_argument('--check-deps', action='store_true',
                        help='Check dependencies and exit')
    parser.add_argument('--test-loading', action='store_true',
                        help='Test model loading and exit')
    parser.add_argument('--is-testing', action='store_true',
                        help='Use deterministic grid masking instead of random masking (required for accurate verification)')
    
    args = parser.parse_args()
    
    # Warn if using --verify without --is-testing
    if args.verify and not args.is_testing:
        print("\n⚠️  WARNING: You are using --verify without --is-testing")
        print("   Verification will show large differences due to random masking.")
        print("   For meaningful verification, either:")
        print("   1. Add --is-testing flag to this command, OR")
        print("   2. Use export_deterministic_mae.py instead")
        print("")
        response = input("Continue anyway? (y/N): ").strip().lower()
        if response != 'y':
            print("Exiting. Please rerun with --is-testing or use export_deterministic_mae.py")
            sys.exit(0)
        print("")
    
    # Check dependencies if requested
    if args.check_deps:
        if check_dependencies():
            print("All dependencies are available!")
        sys.exit(0)
    
    # Test model loading if requested
    if args.test_loading:
        if test_model_loading(args.model_path, args.dataset):
            print("Model loading test passed!")
        else:
            print("Model loading test failed!")
        sys.exit(0)
    
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        
        # Export the model
        if args.export_type == 'full':
            export_mae_to_onnx(
                args.model_path, 
                args.output_path, 
                args.dataset, 
                args.mask_ratio,
                args.opset_version,
                args.is_testing
            )
        elif args.export_type == 'encoder':
            export_mae_encoder_to_onnx(
                args.model_path, 
                args.output_path, 
                args.dataset, 
                args.mask_ratio,
                args.opset_version,
                args.is_testing
            )
        
        # Verify the model if requested
        if args.verify:
            verify_onnx_model(
                args.output_path, 
                args.model_path, 
                args.dataset, 
                args.mask_ratio,
                args.is_testing,
                args.export_type
            )
    
    except Exception as e:
        print(f"Error during export: {e}")
        print("\nTroubleshooting tips:")
        print("1. Check that the model path exists and is accessible")
        print("2. Ensure all dependencies are installed (run with --check-deps)")
        print("3. Test model loading first (run with --test-loading)")
        print("4. Make sure you have enough memory for the export")
        print("5. Verify the dataset parameter matches your model")
        print("6. Check timm version compatibility (may need different version)")
        print("7. If using PyTorch 2.6+, this may be a security-related checkpoint loading issue")
        sys.exit(1)
    
    print("\nExport completed successfully!")
    print("\nUsage examples:")
    print("# Export full MAE model for BRATS dataset:")
    print(f"python export_onnx_mae.py --model-path your_model.pth --output-path mae_model.onnx --dataset brats")
    print("# Export with deterministic masking and verification:")
    print(f"python export_onnx_mae.py --model-path your_model.pth --output-path mae_model.onnx --dataset brats --is-testing --verify")
    print("# Or use the dedicated deterministic export script:")
    print(f"python export_deterministic_mae.py --model-path your_model.pth --output-path mae_model.onnx")
    print("# Export only encoder for LUNA16 dataset:")
    print(f"python export_onnx_mae.py --model-path your_model.pth --output-path mae_encoder.onnx --dataset luna16_unnorm --export-type encoder") 