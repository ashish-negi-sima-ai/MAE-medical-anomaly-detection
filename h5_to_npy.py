#!/usr/bin/env python3
"""
Convert H5 files back to .npy format matching the original pipeline.
Creates directory structure expected by extract_reconstructions.py
"""

import argparse
import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm


def convert_h5_to_npy(h5_path, modality_idx=0):
    """
    Convert H5 file to .npy format and return data + tumor status.
    
    Args:
        h5_path: Path to H5 file
        modality_idx: Which modality to extract (0=T1, 1=T1ce, 2=T2, 3=FLAIR)
    
    Returns:
        npy_data: numpy array ready to save
        has_tumor: boolean indicating tumor presence
    """
    with h5py.File(h5_path, 'r') as f:
        # Load image (240, 240, 4)
        image = f['image'][:]
        
        # Extract single modality
        image_single = image[:, :, modality_idx]
        
        # Check if it has tumor mask
        if 'mask' in f:
            mask = f['mask'][:]
            has_tumor = mask.sum() > 0
        else:
            has_tumor = False
    
    # Save as .npy with shape (240, 240, 1) to match original format
    npy_data = np.expand_dims(image_single, axis=2).astype(np.float32)
    
    return npy_data, has_tumor


def main():
    parser = argparse.ArgumentParser(
        description='Convert H5 files to .npy format for extract_reconstructions.py'
    )
    parser.add_argument('--h5-dir', type=str, required=True,
                        help='Directory containing H5 files')
    parser.add_argument('--output-base', type=str, required=True,
                        help='Base output directory (will create normal/abnormal subdirs)')
    parser.add_argument('--modality', type=int, default=1,
                        help='Modality: 0=T1, 1=T1ce, 2=T2, 3=FLAIR (default: 1)')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Data split name')
    
    args = parser.parse_args()
    
    h5_dir = Path(args.h5_dir)
    output_base = Path(args.output_base)
    
    # Create output directories
    normal_dir = output_base / args.split / 'normal'
    abnormal_dir = output_base / args.split / 'abnormal'
    normal_dir.mkdir(parents=True, exist_ok=True)
    abnormal_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("H5 → NPY Converter")
    print(f"{'='*60}")
    print(f"Input: {h5_dir}")
    print(f"Output: {output_base}")
    print(f"Split: {args.split}")
    print(f"Modality: {args.modality} (T1ce)")
    print(f"{'='*60}\n")
    
    # Get all H5 files
    h5_files = sorted(h5_dir.glob('*.h5'))
    print(f"Found {len(h5_files)} H5 files\n")
    
    normal_count = 0
    abnormal_count = 0
    
    for h5_file in tqdm(h5_files, desc="Converting"):
        # Convert and get tumor status
        npy_data, has_tumor = convert_h5_to_npy(h5_file, args.modality)
        
        # Save to appropriate directory
        output_dir = abnormal_dir if has_tumor else normal_dir
        output_file = output_dir / f"{Path(h5_file).stem}.npy"
        np.save(output_file, npy_data)
        
        if has_tumor:
            abnormal_count += 1
        else:
            normal_count += 1
    
    print(f"\n{'='*60}")
    print("Conversion Complete!")
    print(f"{'='*60}")
    print(f"Normal samples: {normal_count}")
    print(f"Abnormal samples: {abnormal_count}")
    print(f"Total: {normal_count + abnormal_count}")
    print(f"\nOutput structure:")
    print(f"  {output_base}/{args.split}/normal/ ({normal_count} files)")
    print(f"  {output_base}/{args.split}/abnormal/ ({abnormal_count} files)")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()

