"""
Single-image ONNX inference script for medical anomaly detection.

Chains both models in one pass:
  1. MAE reconstruction model  – reconstructs masked patches
  2. Anomaly classifier model  – classifies the reconstruction error

Usage:
    python3 infer_single_onnx.py \
        --input image.npy \
        --mae-model exported_onnx_models/mae_brats_deterministic_grid_masking_simplified.onnx \
        --cls-model exported_onnx_models/classifier_brats_split.onnx \
        --dataset brats
"""

import argparse

import numpy as np
import onnxruntime as ort
from skimage.metrics import structural_similarity as ssim
from skimage.transform import resize

PATCH_SIZE = 16

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_onnx_model(model_path):
    """Load an ONNX model with ONNX Runtime (GPU preferred, CPU fallback)."""
    print(f"Loading ONNX model: {model_path}")
    session = ort.InferenceSession(
        model_path,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    print(f"  providers: {session.get_providers()}")
    return session


def get_deterministic_mask(num_patches, seed=42):
    """1-in-4 grid mask identical to the one baked into the ONNX export."""
    state = np.random.get_state()
    np.random.seed(seed)

    arr = np.random.random(num_patches)
    mask = []
    step = 4
    for i in range(0, len(arr), step):
        max_val = np.max(arr[i : i + step])
        for j in range(i, i + step):
            mask.append(arr[j] != max_val)  # 0 = keep, 1 = remove

    np.random.set_state(state)
    return np.array(mask).astype(np.float32)


# ---------------------------------------------------------------------------
# Stage 1 – MAE reconstruction
# ---------------------------------------------------------------------------

def mae_reconstruct(session, img_nhwc, precomputed_mask):
    """
    Run the MAE model once and return im_paste
    (visible original patches + reconstructed masked patches).

    A single pass suffices because the mask is deterministic — every
    forward pass produces identical output.
    """
    x = np.transpose(img_nhwc, (0, 3, 1, 2)).astype(np.float32)  # NCHW
    input_name = session.get_inputs()[0].name

    outputs = session.run(None, {input_name: x})
    reconstruction = outputs[0]  # (B, 1, H, W)

    # Determine mask
    mask = precomputed_mask
    if len(mask.shape) == 1:
        mask = np.tile(mask, (x.shape[0], 1))

    B, _, H, W = x.shape
    grid_h = H // PATCH_SIZE
    grid_w = W // PATCH_SIZE

    # Expand patch-level mask → pixel-level mask
    mask_px = mask.reshape(B, grid_h, grid_w)
    mask_px = mask_px[:, :, :, np.newaxis, np.newaxis]
    mask_px = np.tile(mask_px, (1, 1, 1, PATCH_SIZE, PATCH_SIZE))
    mask_px = mask_px.transpose(0, 1, 3, 2, 4).reshape(B, H, W)
    mask_px = mask_px[:, np.newaxis, :, :]  # (B, 1, H, W)

    im_paste = x * (1 - mask_px) + reconstruction * mask_px
    return np.transpose(im_paste, (0, 2, 3, 1))  # back to NHWC


# ---------------------------------------------------------------------------
# Stage 2 – Anomaly classification
# ---------------------------------------------------------------------------

def classify(session, diff_nhwc):
    """
    Feed the |original − reconstruction| diff image through the classifier.
    Returns (anomaly_score, label).
      score  – probability of being *normal* (lower → more anomalous)
      label  – 'normal' or 'anomalous'
    """
    x = np.transpose(diff_nhwc, (0, 3, 1, 2)).astype(np.float32)  # NCHW
    input_name = session.get_inputs()[0].name
    logits = session.run(None, {input_name: x})[0]  # (B, 2)

    # Stable softmax
    exp_x = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = exp_x / np.sum(exp_x, axis=1, keepdims=True)

    normal_prob = float(probs[0, 0])
    anomalous_prob = float(probs[0, 1])
    label = "normal" if normal_prob > anomalous_prob else "anomalous"
    return normal_prob, anomalous_prob, label


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Single-image ONNX anomaly detection inference"
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Path to a single .npy image file")
    parser.add_argument("--mae-model", type=str, required=True,
                        help="Path to the MAE reconstruction ONNX model")
    parser.add_argument("--cls-model", type=str, required=True,
                        help="Path to the anomaly classifier ONNX model")
    parser.add_argument("--dataset", type=str, default="brats",
                        choices=["brats", "luna16_unnorm"],
                        help="Dataset type (affects resize & normalization)")
    parser.add_argument("--visualize", action="store_true",
                        help="Show original, reconstruction, and diff images")
    args = parser.parse_args()

    # --- Load models ---
    mae_session = load_onnx_model(args.mae_model)
    cls_session = load_onnx_model(args.cls_model)

    # --- Load & preprocess NPY image ---
    image = np.float32(np.load(args.input))
    if args.dataset == "brats":
        image = image[:, :, 0]  # take first channel
    image = np.expand_dims(image, axis=2)  # (H, W, 1)

    if args.dataset == "brats":
        image = resize(image, (224, 224), order=3)
    else:
        image = resize(image, (64, 64), order=3)

    # Normalize
    if args.dataset == "brats":
        mean, std = 0.0, 1.0
    else:  # luna16_unnorm
        mean, std = 0.0, 100.0
    image = (image - mean) / std

    img_batch = np.expand_dims(image, axis=0).astype(np.float32)  # (1, H, W, 1)

    # --- Stage 1: MAE reconstruction ---
    img_size = 224 if args.dataset == "brats" else 64
    num_patches = (img_size // PATCH_SIZE) ** 2
    mask = get_deterministic_mask(num_patches, seed=42)

    recon = mae_reconstruct(mae_session, img_batch, mask)
    # recon shape: (1, H, W, 1)

    # --- Stage 2: classify the diff ---
    diff = np.abs(img_batch - recon)
    ssim_val = ssim(img_batch[0, :, :, 0], recon[0, :, :, 0], data_range=1.0)

    normal_prob, anomalous_prob, label = classify(cls_session, diff)

    # --- Report ---
    print("\n===== Inference Result =====")
    print(f"  Input       : {args.input}")
    print(f"  SSIM        : {ssim_val:.4f}")
    print(f"  Normal prob : {normal_prob:.4f}")
    print(f"  Anomaly prob: {anomalous_prob:.4f}")
    print(f"  Prediction  : {label}")
    print("============================\n")

    # --- Optional visualisation ---
    if args.visualize:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(img_batch[0, :, :, 0], cmap="gray")
        axes[0].set_title("Original")
        axes[1].imshow(recon[0, :, :, 0], cmap="gray")
        axes[1].set_title("MAE Reconstruction")
        axes[2].imshow(diff[0, :, :, 0], cmap="gray")
        axes[2].set_title(f"Diff  |  {label} ({anomalous_prob:.2f})")
        for ax in axes:
            ax.axis("off")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()

