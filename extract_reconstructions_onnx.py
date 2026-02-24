"""
ONNX Runtime version of extract_reconstructions.py

Replaces PyTorch model inference with ONNX Runtime for the pretrained MAE model.
Usage:
    python3 extract_reconstructions_onnx.py \
        --dataset=brats --mask-ratio=0.75 \
        --model-path=exported_onnx_models/mae_brats_deterministic_grid_masking_simplified.onnx \
        --batch-size=1 --num-trials=4 \
        --output-folder=output_folder_onnx --test
"""

import argparse
import glob
import os.path
import pickle

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
from skimage.transform import resize
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS = ['brats', 'luna16_unnorm']

# Patch size used by the MAE model
PATCH_SIZE = 16


def load_onnx_model(model_path):
    """Load ONNX model with onnxruntime."""
    print(f"Loading ONNX model from: {model_path}")
    session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    print(f"Using providers: {session.get_providers()}")
    return session


def get_deterministic_mask(num_patches, seed=42):
    """
    Generate the exact same 1-in-4 grid mask used in models_mae.py
    and export_deterministic_mae.py during ONNX export.
    """
    # Store current state to avoid affecting other random operations
    state = np.random.get_state()
    np.random.seed(seed)

    arr = np.random.random(num_patches)
    mask = []
    step = 4
    for index_ in range(0, len(arr), step):
        max_ = np.max(arr[index_: index_ + step])
        for index_2 in range(index_, index_ + step):
            mask.append(arr[index_2] != max_)  # 0 keep, 1 remove

    # Restore random state
    np.random.set_state(state)
    return np.array(mask).astype(np.float32)


def apply_gaussian_filter(image):
    sigma = np.random.random() * 3 + 2
    image = cv.GaussianBlur(image, (5, 5), sigma)
    return image


def degrade_reconstruction(img_, apply_blur):
    if args.dataset == 'brats':
        h = np.random.randint(10, 40)  # max H
        w = np.random.randint(10, 40)  # max w
    elif args.dataset == 'luna16_unnorm':
        h = np.random.randint(5, 12)  # max H
        w = np.random.randint(5, 12)  # max W
    else:
        raise ValueError(f'Dataset {args.dataset} not recognized!')

    start_x = np.random.randint(img_.shape[1] - w)
    start_y = np.random.randint(img_.shape[0] - h)

    end_x = start_x + w
    end_y = start_y + h
    if apply_blur:
        img_[start_y: end_y, start_x: end_x, 0] = apply_gaussian_filter(img_[start_y: end_y, start_x: end_x, 0])
    else:
        ratio = np.random.random()
        img_[start_y: end_y, start_x: end_x] *= ratio
    return img_


def apply_degradation(img_, num_iter, apply_blur):
    for _ in range(num_iter):
        img_ = degrade_reconstruction(img_, apply_blur)
    return img_


def get_reconstructions(session, imgs_, idx, precomputed_mask=None):
    """
    Run ONNX MAE inference and return im_paste (visible patches from original + reconstructed masked patches).

    The ONNX model returns:
        - reconstruction: (B, 1, H, W) - the full reconstructed image (already unpatchified)
        - mask: (B, L) - patch-level mask where 1=masked/removed, 0=kept
    """
    # imgs_ is (B, H, W, C) in NHWC format, convert to NCHW
    x = np.transpose(imgs_, (0, 3, 1, 2)).astype(np.float32)

    # Run ONNX inference
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: x})
    reconstruction = outputs[0]  # (B, 1, H, W) in NCHW

    # Determine which mask to use
    if precomputed_mask is not None:
        mask = precomputed_mask  # Should be (B, L) or (L,)
        if len(mask.shape) == 1:
            mask = np.tile(mask, (x.shape[0], 1))
    elif len(outputs) > 1:
        mask = outputs[1]        # (B, L) where L = num_patches
    else:
        mask = None

    if mask is not None:
        B = x.shape[0]
        H, W = x.shape[2], x.shape[3]
        grid_h = H // PATCH_SIZE
        grid_w = W // PATCH_SIZE

        # Expand patch-level mask to pixel-level: (B, L) -> (B, 1, H, W)
        mask_pixels = mask.reshape(B, grid_h, grid_w)
        mask_pixels = mask_pixels[:, :, :, np.newaxis, np.newaxis]  # (B, grid_h, grid_w, 1, 1)
        mask_pixels = np.tile(mask_pixels, (1, 1, 1, PATCH_SIZE, PATCH_SIZE))  # (B, grid_h, grid_w, p, p)
        # Reshape to (B, H, W): need to interleave grid and patch dims
        mask_pixels = mask_pixels.transpose(0, 1, 3, 2, 4)  # (B, grid_h, p, grid_w, p)
        mask_pixels = mask_pixels.reshape(B, H, W)
        mask_pixels = mask_pixels[:, np.newaxis, :, :]  # (B, 1, H, W)

        # MAE reconstruction pasted with visible patches (same logic as PyTorch version):
        # im_paste = original * (1 - mask) + reconstruction * mask
        im_paste = x * (1 - mask_pixels) + reconstruction * mask_pixels
    else:
        # Fallback if no mask is available
        im_paste = reconstruction

    # Convert back to NHWC
    im_paste = np.transpose(im_paste, (0, 2, 3, 1))
    return im_paste


def get_reconstructions_multi(session, imgs_, precomputed_mask=None):
    num_fwd = args.num_trials
    results = None
    for idx in range(num_fwd):
        result = get_reconstructions(session, imgs_, idx, precomputed_mask=precomputed_mask)
        if results is None:
            results = result
        else:
            results += result

    results = results / num_fwd
    return results


# change it to match your own path.
def get_normal_images_paths_train():

    if args.dataset == 'luna16_unnorm':
        return glob.glob(os.path.join(BASE_DIR, 'dataset/luna16/train_unnorm/normal/*.npy'))
    elif args.dataset == 'brats':
        return glob.glob(os.path.join(BASE_DIR, 'dataset/BraTS2020_training_data/split/train/normal/*.npy'))
    else:
        raise ValueError(f'Data set {args.dataset} not recognized.')

# change it to match your own path.
def get_normal_images_paths():
    if args.dataset == 'luna16_unnorm':
        if args.use_val:
            return glob.glob(os.path.join(BASE_DIR, 'dataset/luna16/val_unnorm/normal/*.npy'))
        else:
            return glob.glob(os.path.join(BASE_DIR, 'dataset/luna16/test_unnorm/normal/*.npy'))
    elif args.dataset == 'brats':
        if args.use_val:
            return glob.glob(os.path.join(BASE_DIR, 'dataset/BraTS2020_training_data/split/val/normal/*.npy'))
        else:
            return glob.glob(os.path.join(BASE_DIR, 'dataset/BraTS2020_training_data/split/test/normal/*.npy'))
    else:
        raise ValueError(f'Data set {args.dataset} not recognized.')


# change it to match your own path.
def get_abnormal_images_paths():
    if args.dataset == 'luna16_unnorm':
        if args.use_val:
            return glob.glob(os.path.join(BASE_DIR, 'dataset/luna16/val_unnorm/abnormal/*.npy'))
        else:
            return glob.glob(os.path.join(BASE_DIR, 'dataset/luna16/test_unnorm/abnormal/*.npy'))
    elif args.dataset == 'brats':
        if args.use_val:
            return glob.glob(os.path.join(BASE_DIR, 'dataset/BraTS2020_training_data/split/val/abnormal/*.npy'))
        else:
            return glob.glob(os.path.join(BASE_DIR, 'dataset/BraTS2020_training_data/split/test/abnormal/*.npy'))
    else:
        raise ValueError(f'Data set {args.dataset} not recognized.')


def load_image(img_path_):
    image_np = np.float32(np.load(img_path_))

    if args.dataset == 'brats':
        image_np = image_np[:, :, 0]
    image_np = np.expand_dims(image_np, axis=2)
    return image_np


def process_image(img_):
    if args.dataset == 'brats':
        mean_ = np.array([0.])
        std_ = np.array([1.])
    elif args.dataset == 'luna16_unnorm':
        mean_ = np.array([0.])
        std_ = np.array([100.])
    else:
        raise ValueError(f'Data set {args.dataset} not recognized.')
    img_ = img_ - mean_
    img_ = img_ / std_
    return img_


def visualize(imgs_, reconstructions_, old_reconstructions_, paths_):
    num_imgs = 5
    for (img_, recon_, old_recon_, path_) in zip(imgs_, reconstructions_, old_reconstructions_, paths_):

        plt.subplot(1, num_imgs, 1)
        plt.imshow(img_, cmap='gray')
        plt.subplot(1, num_imgs, 2)
        plt.imshow(recon_, cmap='gray')
        plt.subplot(1, num_imgs, 3)
        plt.imshow(old_recon_, cmap='gray')
        plt.subplot(1, num_imgs, 4)
        plt.imshow(np.abs(img_ - recon_), cmap='gray')
        plt.subplot(1, num_imgs, 5)
        plt.imshow(np.abs(old_recon_ - recon_), cmap='gray')
        plt.show()


def save(imgs, reconstructions, used_paths, is_abnormal, iter_):
    base_dir = args.output_folder
    if is_abnormal:
        base_dir = os.path.join(base_dir, 'abnormal')
    else:
        base_dir = os.path.join(base_dir, 'normal')

    for (img_, recon_, path_) in zip(imgs, reconstructions, used_paths):
        if is_abnormal and img_.sum() == 0:
            continue

        info_ = {'img': img_, 'recon': recon_}
        short_filename = os.path.split(path_)[-1][:-4] + f'_{iter_}.pkl'
        with open(os.path.join(base_dir, short_filename), 'wb') as handle:
            pickle.dump(info_, handle)


def write_reconstructions(session, paths, is_abnormal: bool = False, iter_: int = 0):

    # Precompute mask if it's supposed to be deterministic
    # The image size depends on the dataset
    img_size = 224 if args.dataset == 'brats' else 64
    num_patches = (img_size // PATCH_SIZE) ** 2
    precomputed_mask = get_deterministic_mask(num_patches, seed=42)

    for start_index in tqdm(range(0, len(paths), args.batch_size)):
        imgs = []
        used_paths = []
        for idx_path in range(start_index, start_index + args.batch_size):
            if idx_path < len(paths):
                path_ = paths[idx_path]
                img_ = load_image(path_)
                if args.dataset == 'brats':
                    img_ = resize(img_, (224, 224), order=3)  # 3: Bi-cubic
                else:
                    img_ = resize(img_, (64, 64), order=3)  # 3: Bi-cubic
                img_ = process_image(img_)

                imgs.append(img_)
                used_paths.append(path_)

        imgs = np.array(imgs, np.float32)

        old_reconstructions = get_reconstructions_multi(session, imgs, precomputed_mask=precomputed_mask)
        if is_abnormal and args.test is False:
            reconstructions = [apply_degradation(recon.copy(), num_iter=np.random.randint(1, 10), apply_blur=False) for recon in old_reconstructions]
        else:
            reconstructions = old_reconstructions

        save(imgs, reconstructions, used_paths, is_abnormal, iter_=iter_)


parser = argparse.ArgumentParser(description='ONNX Medical Images Reconstruction Extraction')
parser.add_argument('--model-path', type=str, required=True,
                    help='Path to ONNX model file (e.g. exported_onnx_models/mae_brats_random.onnx)')
parser.add_argument('--mask-ratio', type=float)
parser.add_argument('--dataset', type=str)
parser.add_argument('--batch-size', type=int, default=1)
parser.add_argument('--output-folder', type=str, required=True)
parser.add_argument('--num-trials', type=int, default=1)
parser.add_argument('--use_val', action='store_true',
                    help='Test on val data.')

parser.add_argument('--test', action='store_true')

parser.set_defaults(use_val=False)

args = parser.parse_args()

assert args.dataset in DATASETS

"""
Example usage:

python3 extract_reconstructions_onnx.py --dataset=brats --mask-ratio=0.75 \
    --model-path=exported_onnx_models/mae_brats_random.onnx \
    --batch-size=64 --num-trials=4 \
    --output-folder=output_folder_onnx --test
"""
if __name__ == '__main__':
    os.makedirs(os.path.join(args.output_folder, 'normal'), exist_ok=True)
    os.makedirs(os.path.join(args.output_folder, 'abnormal'), exist_ok=True)

    onnx_session = load_onnx_model(args.model_path)

    # Data
    if args.test:
        write_reconstructions(onnx_session, paths=get_normal_images_paths(), is_abnormal=False)
        write_reconstructions(onnx_session, paths=get_abnormal_images_paths(), is_abnormal=True)
    else:
        normal_paths = get_normal_images_paths_train()
        write_reconstructions(onnx_session, paths=normal_paths, is_abnormal=False)
        write_reconstructions(onnx_session, paths=normal_paths, is_abnormal=True, iter_=1)
