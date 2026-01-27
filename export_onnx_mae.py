import argparse
import os
import sys

# Check for required dependencies
try:
    import torch
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


def prepare_model(chkpt_dir, arch='mae_vit_large_patch16'):
    """Prepare and load the MAE model from checkpoint"""
    # build model
    img_size = 64
    patch_size = 16
    if args.dataset == 'brats':
        img_size = 224
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
    """Wrapper class to make MAE model ONNX-compatible"""
    
    def __init__(self, mae_model, mask_ratio=0.75, is_testing=False):
        super().__init__()
        self.mae_model = mae_model
        self.mask_ratio = mask_ratio
        self.is_testing = is_testing
    
    def forward(self, x):
        # Forward pass with fixed mask ratio
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
        verbose=True
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
        verbose=True
    )
    
    print(f"Successfully exported MAE encoder to {output_path}")


def verify_onnx_model(onnx_path, dataset):
    """Verify the exported ONNX model"""
    try:
        import onnx
        import onnxruntime as ort
        
        # Load ONNX model
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model validation passed!")
        
        # Test inference with ONNX Runtime
        ort_session = ort.InferenceSession(onnx_path)
        
        # Prepare test input
        if dataset == 'brats':
            test_input = np.random.randn(1, 1, 224, 224).astype(np.float32)
        else:
            test_input = np.random.randn(1, 1, 64, 64).astype(np.float32)
        
        # Run inference
        ort_inputs = {ort_session.get_inputs()[0].name: test_input}
        ort_outputs = ort_session.run(None, ort_inputs)
        
        print(f"ONNX Runtime inference successful!")
        print(f"Input shape: {test_input.shape}")
        for i, output in enumerate(ort_outputs):
            print(f"Output {i} shape: {output.shape}")
            
    except ImportError:
        print("ONNX or ONNXRuntime not installed. Skipping verification.")
        print("To verify the model, install: pip install onnx onnxruntime")
    except Exception as e:
        print(f"ONNX model verification failed: {e}")


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
                        help='Verify the exported ONNX model')
    parser.add_argument('--check-deps', action='store_true',
                        help='Check dependencies and exit')
    parser.add_argument('--test-loading', action='store_true',
                        help='Test model loading and exit')
    parser.add_argument('--is-testing', action='store_true',
                        help='Use grid masking instead of random masking')
    
    args = parser.parse_args()
    
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
            verify_onnx_model(args.output_path, args.dataset)
    
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
    print(f"python export_onnx.py --model-path your_model.pth --output-path mae_model.onnx --dataset brats")
    print("# Export only encoder for LUNA16 dataset:")
    print(f"python export_onnx.py --model-path your_model.pth --output-path mae_encoder.onnx --dataset luna16_unnorm --export-type encoder") 