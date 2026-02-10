#!/usr/bin/env python3
"""
Simple inference script for MAE-based medical anomaly detection.
Takes a single NPY file, performs reconstruction, and classifies it.
"""

import argparse
import numpy as np
import torch
from skimage.transform import resize
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
from functools import partial

import models_mae
import models_vit


def load_mae_model(model_path, img_size=224, patch_size=16):
    """Load the pretrained MAE model for reconstruction."""
    model = models_mae.mae_vit_base_patch16_dec512d8b(
        img_size=img_size, 
        patch_size=patch_size
    )
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(model_path, map_location=device)
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(f"MAE Model loaded: {msg}")
    
    model = model.to(device)
    model.eval()
    return model, device


def load_classifier_model(model_path, img_size=224):
    """Load the finetuned classifier model."""
    model = models_vit.vit_base_patch16(
        num_classes=2,
        drop_path_rate=0.0,
        global_pool=True,
        img_size=img_size
    )
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(model_path, map_location=device)
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(f"Classifier Model loaded: {msg}")
    
    model = model.to(device)
    model.eval()
    return model, device


def load_and_preprocess_image(npy_path, target_size=224):
    """Load NPY file and preprocess it."""
    # Load the numpy array
    image_np = np.float32(np.load(npy_path))
    
    # Extract first channel (modality) if multiple channels
    if len(image_np.shape) == 3:
        image_np = image_np[:, :, 0]
    
    # Add channel dimension
    image_np = np.expand_dims(image_np, axis=2)
    
    # Resize to target size
    image_np = resize(image_np, (target_size, target_size), order=3)
    
    # Normalize (mean=0, std=1 for BraTS)
    mean_ = np.array([0.])
    std_ = np.array([1.])
    image_np = (image_np - mean_) / std_
    
    return image_np


def reconstruct_image(mae_model, image, device, mask_ratio=0.75, num_trials=4):
    """Perform MAE reconstruction with multiple trials."""
    # Add batch dimension
    x = torch.tensor(image[np.newaxis, ...])
    
    # Change from NHWC to NCHW
    x = torch.einsum('nhwc->nchw', x)
    x = x.to(device).float()
    
    # Multiple reconstruction trials
    reconstructions = []
    with torch.no_grad():
        for idx in range(num_trials):
            loss, result, mask = mae_model(x, mask_ratio=mask_ratio, idx_masking=idx, is_testing=False)
            result = mae_model.unpatchify(result)
            result = torch.einsum('nchw->nhwc', result).detach().cpu()
            
            # Visualize the mask
            mask = mask.detach()
            mask = mask.unsqueeze(-1).repeat(1, 1, mae_model.patch_embed.patch_size[0]**2 * 1)
            mask = mae_model.unpatchify(mask)  # 1 is removing, 0 is keeping
            mask = torch.einsum('nchw->nhwc', mask).detach().cpu()
            
            # CRITICAL: Paste reconstruction with visible patches
            # This is what the original code does - hybrid of real + reconstructed patches
            im_paste = torch.einsum('nchw->nhwc', x).detach().cpu() * (1 - mask) + result * mask
            
            reconstructions.append(im_paste.numpy())
    
    # Average the reconstructions
    avg_reconstruction = np.mean(reconstructions, axis=0)[0]
    
    return avg_reconstruction


def classify_difference(classifier_model, diff_image, device):
    """Classify the difference image to get anomaly score.
    
    NOTE: The model was trained using ImageFolder which assigns labels alphabetically:
    - Class 0 = abnormal (first alphabetically)
    - Class 1 = normal (second alphabetically)
    """
    # Normalize the difference
    mean_ = np.array([0.])
    std_ = np.array([1.])
    diff_normalized = (diff_image - mean_) / std_
    
    # Add batch dimension
    x = torch.tensor(diff_normalized[np.newaxis, ...])
    
    # Change from NHWC to NCHW
    x = torch.einsum('nhwc->nchw', x)
    x = x.to(device).float()
    
    # Get classification scores
    with torch.no_grad():
        result = classifier_model(x)
        soft_result = torch.nn.functional.softmax(result, dim=1)
    
    # Model outputs probabilities for [class_0, class_1]
    # Based on ImageFolder: class_0=abnormal, class_1=normal
    # Return as [normal, abnormal] for display
    probs = soft_result.detach().cpu().numpy()[0]
    # probs[0] = P(class_0) = P(abnormal)
    # probs[1] = P(class_1) = P(normal)
    # BUT: Testing shows we should NOT swap! Return as-is and interpret differently
    return probs  # [P(abnormal), P(normal)] - will reinterpret in display


def visualize_results(original, reconstruction, difference, scores, ssim_score, save_path=None):
    """Visualize the original image, reconstruction, and difference."""
    import os
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    axes[0].imshow(original[:, :, 0], cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(reconstruction[:, :, 0], cmap='gray')
    axes[1].set_title('Reconstruction')
    axes[1].axis('off')
    
    axes[2].imshow(difference[:, :, 0], cmap='hot')
    axes[2].set_title(f'Difference (SSIM: {ssim_score:.4f})')
    axes[2].axis('off')
    
    # Classification result
    axes[3].bar(['Normal', 'Abnormal'], scores, color=['green', 'red'])
    axes[3].set_title('Classification Scores')
    axes[3].set_ylim([0, 1])
    axes[3].set_ylabel('Probability')
    
    plt.tight_layout()
    
    if save_path:
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(save_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Simple MAE inference for medical anomaly detection')
    parser.add_argument('--input', type=str, required=True, help='Path to input NPY file')
    parser.add_argument('--mae-model', type=str, default='models/brats_pretrained.pth',
                        help='Path to pretrained MAE model')
    parser.add_argument('--classifier-model', type=str, default='models/brats_finetuned.pth',
                        help='Path to finetuned classifier model')
    parser.add_argument('--mask-ratio', type=float, default=0.75,
                        help='Masking ratio for MAE reconstruction')
    parser.add_argument('--num-trials', type=int, default=4,
                        help='Number of reconstruction trials')
    parser.add_argument('--img-size', type=int, default=224,
                        help='Image size')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for visualization (if not provided, will display)')
    parser.add_argument('--no-viz', action='store_true',
                        help='Skip visualization')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("MAE Medical Anomaly Detection - Simple Inference")
    print("=" * 60)
    
    # Step 1: Load models
    print(f"\n[1/5] Loading MAE model from: {args.mae_model}")
    mae_model, mae_device = load_mae_model(args.mae_model, img_size=args.img_size)
    
    print(f"\n[2/5] Loading Classifier model from: {args.classifier_model}")
    classifier_model, clf_device = load_classifier_model(args.classifier_model, img_size=args.img_size)
    
    # Step 2: Load and preprocess image
    print(f"\n[3/5] Loading and preprocessing image: {args.input}")
    original_image = load_and_preprocess_image(args.input, target_size=args.img_size)
    print(f"Image shape: {original_image.shape}")
    
    # Step 3: Reconstruct image
    print(f"\n[4/5] Performing MAE reconstruction (mask_ratio={args.mask_ratio}, trials={args.num_trials})...")
    reconstruction = reconstruct_image(mae_model, original_image, mae_device, 
                                       mask_ratio=args.mask_ratio, 
                                       num_trials=args.num_trials)
    
    # Compute difference
    difference = np.abs(original_image - reconstruction)
    
    # Compute SSIM
    data_range = max(original_image[:, :, 0].max(), reconstruction[:, :, 0].max()) - \
                 min(original_image[:, :, 0].min(), reconstruction[:, :, 0].min())
    if data_range == 0:
        data_range = 1.0
    ssim_score = ssim(original_image[:, :, 0], reconstruction[:, :, 0], data_range=data_range)
    
    # Step 4: Classify
    print(f"\n[5/5] Classifying reconstruction error...")
    classification_scores = classify_difference(classifier_model, difference, clf_device)
    
    # Results
    # classification_scores are [P(abnormal), P(normal)] from model
    # Swap for display
    normal_prob = classification_scores[1]
    abnormal_prob = classification_scores[0]
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"SSIM Score: {ssim_score:.4f} (higher = more similar)")
    print(f"Normal Probability: {normal_prob:.4f}")
    print(f"Abnormal Probability: {abnormal_prob:.4f}")
    print(f"\nPrediction: {'NORMAL' if normal_prob > abnormal_prob else 'ABNORMAL'}")
    print("=" * 60)
    
    # Visualization
    # Always generate if output path is provided, otherwise respect no_viz flag
    if args.output or not args.no_viz:
        if not args.no_viz:
            print("\n[6/6] Generating visualization...")
        # For visualization, pass as [normal, abnormal]
        viz_scores = np.array([normal_prob, abnormal_prob])
        visualize_results(original_image, reconstruction, difference, 
                         viz_scores, ssim_score, save_path=args.output)
    
    return {
        'ssim': ssim_score,
        'normal_prob': float(normal_prob),
        'abnormal_prob': float(abnormal_prob),
        'prediction': 'NORMAL' if normal_prob > abnormal_prob else 'ABNORMAL'
    }


if __name__ == '__main__':
    main()

