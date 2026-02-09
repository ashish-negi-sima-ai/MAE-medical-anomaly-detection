#!/usr/bin/env python3
"""
Test script to verify MAE Medical Anomaly Detection installation
Run this after setting up your environment to ensure everything works correctly.
"""

import sys

def test_imports():
    """Test all required imports"""
    print("="*60)
    print("Testing Package Imports")
    print("="*60)
    
    tests_passed = 0
    tests_failed = 0
    
    # Core ML libraries
    packages = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("timm", "timm (PyTorch Image Models)"),
        ("numpy", "NumPy"),
        ("PIL", "Pillow"),
        ("cv2", "OpenCV"),
        ("skimage", "scikit-image"),
        ("albumentations", "Albumentations"),
        ("h5py", "h5py"),
        ("nibabel", "NiBabel"),
        ("sklearn", "scikit-learn"),
        ("matplotlib", "Matplotlib"),
        ("tqdm", "tqdm"),
    ]
    
    for module_name, display_name in packages:
        try:
            module = __import__(module_name)
            version = getattr(module, "__version__", "unknown")
            print(f"✓ {display_name:30s} version: {version}")
            tests_passed += 1
        except ImportError as e:
            print(f"✗ {display_name:30s} FAILED: {e}")
            tests_failed += 1
    
    print()
    return tests_passed, tests_failed


def test_timm_version():
    """Test that timm is the correct version (0.3.2)"""
    print("="*60)
    print("Testing timm Version")
    print("="*60)
    
    try:
        import timm
        version = timm.__version__
        if version == "0.3.2":
            print(f"✓ timm version is correct: {version}")
            return 1, 0
        else:
            print(f"✗ timm version is INCORRECT: {version} (expected 0.3.2)")
            print("  Please run: pip uninstall timm -y && pip install timm==0.3.2")
            return 0, 1
    except ImportError:
        print("✗ timm not installed")
        return 0, 1
    finally:
        print()


def test_pytorch_cuda():
    """Test PyTorch CUDA availability"""
    print("="*60)
    print("Testing PyTorch CUDA Support")
    print("="*60)
    
    try:
        import torch
        
        print(f"PyTorch Version: {torch.__version__}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"cuDNN Version: {torch.backends.cudnn.version()}")
            print(f"Number of GPUs: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
                props = torch.cuda.get_device_properties(i)
                print(f"    Memory: {props.total_memory / 1024**3:.2f} GB")
        else:
            print("⚠️  Warning: CUDA not available. Training will be slow on CPU.")
            print("   For GPU support, reinstall PyTorch with CUDA:")
            print("   conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia")
        
        print()
        return 1, 0
    except Exception as e:
        print(f"✗ PyTorch CUDA test failed: {e}")
        print()
        return 0, 1


def test_model_loading():
    """Test loading MAE model"""
    print("="*60)
    print("Testing MAE Model Loading")
    print("="*60)
    
    try:
        import torch
        import models_mae
        
        # Test model creation
        model = models_mae.mae_vit_base_patch16(img_size=224, patch_size=16)
        print("✓ MAE model created successfully")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        
        # Test forward pass
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        x = torch.randn(2, 1, 224, 224).to(device)
        
        with torch.no_grad():
            loss, pred, mask = model(x, mask_ratio=0.75)
        
        print(f"✓ Forward pass successful")
        print(f"  Input shape: {x.shape}")
        print(f"  Prediction shape: {pred.shape}")
        print(f"  Mask shape: {mask.shape}")
        print(f"  Loss: {loss.item():.4f}")
        
        # Test reconstruction
        reconstruction = model.unpatchify(pred)
        print(f"  Reconstruction shape: {reconstruction.shape}")
        
        print()
        return 1, 0
        
    except Exception as e:
        print(f"✗ Model loading test failed: {e}")
        import traceback
        traceback.print_exc()
        print()
        return 0, 1


def test_vit_model():
    """Test loading ViT model for classification"""
    print("="*60)
    print("Testing ViT Classification Model")
    print("="*60)
    
    try:
        import torch
        import models_vit
        
        # Test model creation
        model = models_vit.vit_base_patch16(
            num_classes=2,
            global_pool=True,
            img_size=224
        )
        print("✓ ViT model created successfully")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Total parameters: {total_params:,}")
        
        # Test forward pass
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        x = torch.randn(2, 1, 224, 224).to(device)
        
        with torch.no_grad():
            output = model(x)
        
        print(f"✓ Forward pass successful")
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {output.shape}")
        
        print()
        return 1, 0
        
    except Exception as e:
        print(f"✗ ViT model test failed: {e}")
        import traceback
        traceback.print_exc()
        print()
        return 0, 1


def test_data_loading():
    """Test basic data loading utilities"""
    print("="*60)
    print("Testing Data Loading Utilities")
    print("="*60)
    
    try:
        import numpy as np
        import h5py
        import tempfile
        import os
        
        # Create a temporary H5 file
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Write test data
            with h5py.File(tmp_path, 'w') as f:
                img_data = np.random.randn(224, 224, 4).astype(np.float32)
                mask_data = np.random.randint(0, 2, (224, 224, 3)).astype(np.uint8)
                
                f.create_dataset('image', data=img_data)
                f.create_dataset('mask', data=mask_data)
            
            print("✓ H5 file write successful")
            
            # Read test data
            with h5py.File(tmp_path, 'r') as f:
                img = np.array(f['image'])
                mask = np.array(f['mask'])
            
            print("✓ H5 file read successful")
            print(f"  Image shape: {img.shape}")
            print(f"  Mask shape: {mask.shape}")
            
        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        
        print()
        return 1, 0
        
    except Exception as e:
        print(f"✗ Data loading test failed: {e}")
        print()
        return 0, 1


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("MAE Medical Anomaly Detection - Installation Test")
    print("="*60)
    print()
    
    total_passed = 0
    total_failed = 0
    
    # Run all tests
    tests = [
        test_imports,
        test_timm_version,
        test_pytorch_cuda,
        test_model_loading,
        test_vit_model,
        test_data_loading,
    ]
    
    for test in tests:
        passed, failed = test()
        total_passed += passed
        total_failed += failed
    
    # Summary
    print("="*60)
    print("Test Summary")
    print("="*60)
    print(f"Tests Passed: {total_passed}")
    print(f"Tests Failed: {total_failed}")
    print()
    
    if total_failed == 0:
        print("✓ All tests passed! Your environment is set up correctly.")
        print()
        print("Next steps:")
        print("  1. Download datasets (BraTS2020, LUNA16)")
        print("  2. Run pretraining: python main_pretrain.py --help")
        print("  3. See QUICKSTART.md for training examples")
        return 0
    else:
        print("✗ Some tests failed. Please fix the issues above.")
        print()
        print("Common fixes:")
        print("  - Missing packages: pip install -r requirements.txt")
        print("  - Wrong timm version: pip install timm==0.3.2")
        print("  - CUDA issues: Reinstall PyTorch with correct CUDA version")
        print()
        print("See SETUP.md for detailed troubleshooting.")
        return 1


if __name__ == "__main__":
    sys.exit(main())


