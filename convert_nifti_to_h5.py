#!/usr/bin/env python3
"""
Convert BraTS20 NIfTI files to H5 format for testing.
Matches the format of test_data/volume_100_slice_108.h5
"""

import argparse
import h5py
import nibabel as nib
import numpy as np
from pathlib import Path
from tqdm import tqdm
import os


def load_nifti_volume(case_dir):
    """
    Load all modalities and segmentation for a BraTS case.
    
    Args:
        case_dir: Path to case directory (e.g., BraTS20_Training_001)
    
    Returns:
        image: (H, W, D, 4) - 4 modalities
        seg: (H, W, D) - segmentation mask (only for training data)
    """
    case_path = Path(case_dir)
    case_name = case_path.name
    
    # Load modalities
    modalities = ['t1', 't1ce', 't2', 'flair']
    volumes = []
    
    for mod in modalities:
        # Try both .nii and .nii.gz
        file_path = case_path / f"{case_name}_{mod}.nii.gz"
        if not file_path.exists():
            file_path = case_path / f"{case_name}_{mod}.nii"
        if not file_path.exists():
            raise FileNotFoundError(f"Missing {mod} file: {file_path}")
        
        nii = nib.load(str(file_path))
        data = nii.get_fdata()
        volumes.append(data)
    
    # Stack modalities: (H, W, D, 4)
    image = np.stack(volumes, axis=-1)
    
    # Load segmentation if available (training data only)
    seg_path = case_path / f"{case_name}_seg.nii.gz"
    if not seg_path.exists():
        seg_path = case_path / f"{case_name}_seg.nii"
    
    if seg_path.exists():
        seg_nii = nib.load(str(seg_path))
        seg = seg_nii.get_fdata()
    else:
        seg = None
    
    return image, seg


def normalize_volume(volume):
    """
    Z-score normalization per modality.
    Matches the preprocessing in test_data h5 files.
    """
    normalized = np.zeros_like(volume, dtype=np.float32)
    
    # Normalize each modality independently
    for i in range(volume.shape[-1]):
        modality = volume[..., i]
        
        # Only normalize non-zero region (brain mask)
        mask = modality > 0
        if mask.sum() > 0:
            mean = modality[mask].mean()
            std = modality[mask].std()
            
            if std > 0:
                normalized[..., i] = np.where(
                    mask,
                    (modality - mean) / std,
                    0
                )
            else:
                normalized[..., i] = modality
    
    return normalized


def convert_segmentation_to_mask(seg):
    """
    Convert BraTS segmentation to binary masks.
    
    BraTS labels:
    - 0: Background
    - 1: Necrotic/Core
    - 2: Edema
    - 4: Enhancing Tumor
    
    Returns masks matching test_data format: (H, W, 3)
    - Channel 0: Necrotic/Core (label 1)
    - Channel 1: Edema (label 2)
    - Channel 2: Enhancing (label 4)
    """
    if seg is None:
        return None
    
    H, W, D = seg.shape
    masks = np.zeros((H, W, D, 3), dtype=np.uint8)
    
    masks[..., 0] = (seg == 1).astype(np.uint8)  # Necrotic/Core
    masks[..., 1] = (seg == 2).astype(np.uint8)  # Edema
    masks[..., 2] = (seg == 4).astype(np.uint8)  # Enhancing
    
    return masks


def extract_slices_to_h5(case_dir, output_dir, min_tumor_pixels=50):
    """
    Extract 2D slices from 3D volume and save as H5 files.
    
    Args:
        case_dir: Path to BraTS case directory
        output_dir: Output directory for H5 files
        min_tumor_pixels: Minimum tumor pixels to save slice (default: 50)
    
    Returns:
        num_slices: Number of slices extracted
    """
    case_path = Path(case_dir)
    case_name = case_path.name
    output_path = Path(output_dir)
    
    # Load volume
    image, seg = load_nifti_volume(case_dir)
    
    # DON'T normalize - model expects raw intensity values
    # (Commenting out normalization to match training preprocessing)
    # image_norm = normalize_volume(image)
    image_norm = image.astype(np.float32)  # Keep raw values
    
    # Convert segmentation
    masks = convert_segmentation_to_mask(seg) if seg is not None else None
    
    # Extract slices (axial view - typically slice along axis 2)
    num_slices = image_norm.shape[2]
    saved_count = 0
    
    for slice_idx in range(num_slices):
        # Extract 2D slice
        img_slice = image_norm[:, :, slice_idx, :]  # (240, 240, 4)
        
        # Check if slice has tumor (if segmentation available)
        if masks is not None:
            mask_slice = masks[:, :, slice_idx, :]  # (240, 240, 3)
            tumor_pixels = mask_slice.sum()
            
            # Skip slices with too little tumor
            if tumor_pixels < min_tumor_pixels:
                continue
        else:
            mask_slice = np.zeros((img_slice.shape[0], img_slice.shape[1], 3), dtype=np.uint8)
        
        # Save as H5
        output_file = output_path / f"{case_name}_slice_{slice_idx:03d}.h5"
        
        with h5py.File(output_file, 'w') as f:
            f.create_dataset('image', data=img_slice.astype(np.float32))
            f.create_dataset('mask', data=mask_slice.astype(np.uint8))
        
        saved_count += 1
    
    return saved_count


def process_dataset(data_dir, output_dir, dataset_type='training'):
    """
    Process entire BraTS dataset.
    
    Args:
        data_dir: Path to BraTS20_TrainingData or BraTS20_ValidationData
        output_dir: Output directory for H5 files
        dataset_type: 'training' or 'validation'
    """
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get all case directories
    case_dirs = sorted([d for d in data_path.iterdir() if d.is_dir()])
    
    print(f"\n{'='*60}")
    print(f"Processing {dataset_type} dataset")
    print(f"{'='*60}")
    print(f"Input: {data_dir}")
    print(f"Output: {output_dir}")
    print(f"Cases found: {len(case_dirs)}")
    print(f"{'='*60}\n")
    
    total_slices = 0
    failed_cases = []
    
    for case_dir in tqdm(case_dirs, desc="Processing cases"):
        try:
            num_slices = extract_slices_to_h5(case_dir, output_path)
            total_slices += num_slices
        except Exception as e:
            print(f"\n✗ Failed on {case_dir.name}: {e}")
            failed_cases.append(case_dir.name)
    
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"{'='*60}")
    print(f"Total cases processed: {len(case_dirs)}")
    print(f"Total slices extracted: {total_slices}")
    print(f"Failed cases: {len(failed_cases)}")
    if failed_cases:
        print(f"Failed: {', '.join(failed_cases)}")
    print(f"{'='*60}\n")
    
    return total_slices


def main():
    parser = argparse.ArgumentParser(
        description='Convert BraTS20 NIfTI files to H5 format'
    )
    parser.add_argument('--input-dir', type=str, required=True,
                        help='Path to BraTS20 data directory')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for H5 files')
    parser.add_argument('--dataset-type', type=str, 
                        choices=['training', 'validation', 'both'],
                        default='training',
                        help='Which dataset to process')
    parser.add_argument('--min-tumor-pixels', type=int, default=50,
                        help='Minimum tumor pixels to save slice (default: 50)')
    
    args = parser.parse_args()
    
    input_path = Path(args.input_dir)
    
    if not input_path.exists():
        print(f"Error: Input directory not found: {args.input_dir}")
        return
    
    print("\n" + "="*60)
    print("BraTS20 NIfTI → H5 Converter")
    print("="*60)
    
    # Check if nibabel is installed
    try:
        import nibabel
    except ImportError:
        print("\n✗ Error: nibabel not installed!")
        print("\nPlease install it:")
        print("  pip install nibabel")
        return
    
    total = 0
    
    # Process training data
    if args.dataset_type in ['training', 'both']:
        training_dir = input_path / 'BraTS2020_TrainingData'
        if training_dir.exists():
            output_dir = Path(args.output_dir) / 'training'
            total += process_dataset(training_dir, output_dir, 'training')
        else:
            print(f"Warning: Training directory not found: {training_dir}")
    
    # Process validation data
    if args.dataset_type in ['validation', 'both']:
        validation_dir = input_path / 'BraTS2020_ValidationData'
        if validation_dir.exists():
            output_dir = Path(args.output_dir) / 'validation'
            total += process_dataset(validation_dir, output_dir, 'validation')
        else:
            print(f"Warning: Validation directory not found: {validation_dir}")
    
    print(f"\n✓ Conversion complete! Total slices: {total}\n")


if __name__ == '__main__':
    main()

