"""
ONNX Runtime version of evaluate_sup.py

Replaces PyTorch model inference with ONNX Runtime for the finetuned classifier model.
Usage:
    python3 evaluate_sup_onnx.py         --dataset=brats         --model-path=exported_onnx_models/classifier_brats_split.onnx         --batch-size=1 --output-folder=output_folder_onnx
"""

import argparse
import glob
import os.path
import pickle

import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import roc_auc_score
from sklearn.utils import shuffle
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS = ['brats', 'luna16_unnorm']


def load_onnx_model(model_path):
    """Load ONNX model with onnxruntime."""
    print(f"Loading ONNX model from: {model_path}")
    session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    print(f"Using providers: {session.get_providers()}")
    return session


def get_scores(session, imgs_):
    """
    Run ONNX classifier inference and return anomaly scores.
    """
    # imgs_ is (B, H, W, C), convert to NCHW
    x = np.transpose(imgs_, (0, 3, 1, 2)).astype(np.float32)

    # Run ONNX inference
    input_name = session.get_inputs()[0].name
    result = session.run(None, {input_name: x})[0]  # Logits (B, 2)

    # Apply softmax in numpy
    # softmax(x) = exp(x) / sum(exp(x))
    # Stability trick: subtract max(x) before exp
    exp_x = np.exp(result - np.max(result, axis=1, keepdims=True))
    soft_result = exp_x / np.sum(exp_x, axis=1, keepdims=True)

    # Return probability of class 0 (normal) as the score?
    # Original PyTorch code: return soft_result.detach().cpu().numpy()[:, 0]
    return soft_result[:, 0]


# change it to match your own path.
def get_normal_images_paths():

    if args.dataset == 'luna16_unnorm':
        if args.use_val:
            return glob.glob(os.path.join(BASE_DIR, 'output_folder/val/normal/*.pkl'))
        else:
            return glob.glob(os.path.join(BASE_DIR, 'output_folder/test/normal/*.pkl'))

    elif args.dataset == 'brats':
        if args.use_val:
            return glob.glob(os.path.join(BASE_DIR, 'output_folder/val/normal/*.pkl'))
        else:
            # Check if user provided an output folder or use default
            output_dir = getattr(args, 'output_folder', 'output_folder')
            return glob.glob(os.path.join(BASE_DIR, f'{output_dir}/normal/*.pkl'))
    else:
        raise ValueError(f'Data set {args.dataset} not recognized.')


# change it to match your own path.
def get_abnormal_images_paths():
    if args.dataset == 'luna16_unnorm':
        if args.use_val:
            return glob.glob(os.path.join(BASE_DIR, 'output_folder/val/abnormal/*.pkl'))
        else:
            return glob.glob(os.path.join(BASE_DIR, 'output_folder/test/abnormal/*.pkl'))

    elif args.dataset == 'brats':
        if args.use_val:
            return glob.glob(os.path.join(BASE_DIR, 'output_folder/val/abnormal/*.pkl'))
        else:
            # Check if user provided an output folder or use default
            output_dir = getattr(args, 'output_folder', 'output_folder')
            return glob.glob(os.path.join(BASE_DIR, f'{output_dir}/abnormal/*.pkl'))
    else:
        raise ValueError(f'Data set {args.dataset} not recognized.')


def mae(img1, img2):
    return np.mean(np.abs(img1 - img2))


def mse(img1, img2):
    return np.mean(np.abs(img1 - img2) ** 2)


def load_image(img_path_):
    with open(img_path_, 'rb') as handle:
        dict_ = pickle.load(handle)

    img_ = dict_["img"]
    recon_ = dict_["recon"]
    diff = np.abs((img_ - recon_))[:, :, 0]

    image_np = np.expand_dims(diff, axis=2)
    return image_np, -ssim(img_[:, :, 0], recon_[:, :, 0], data_range=1.0)


def process_image(img_):
    mean_ = np.array([0.])
    std_ = np.array([1.])

    img_ = img_ - mean_
    img_ = img_ / std_
    return img_


def get_auc(session, paths, ground_truth_labels):
    pred_labels = []
    for start_index in tqdm(range(0, len(paths), args.batch_size)):
        imgs = []
        ssim_scores = []
        for idx_path in range(start_index, start_index + args.batch_size):
            if idx_path < len(paths):
                path_ = paths[idx_path]
                img_, ssim_value = load_image(path_)

                img_ = process_image(img_)
                imgs.append(img_)
                ssim_scores.append(ssim_value)

        imgs = np.array(imgs, np.float32)

        scores = get_scores(session, imgs)
        final_scores = scores # + np.array(ssim_scores)

        pred_labels.extend(list(final_scores))

    pred_labels = np.array(pred_labels)

    auc = roc_auc_score(ground_truth_labels, pred_labels)
    print("AUC:", auc)

    return auc


def compare_histogram(scores, classes, thresh=None, n_bins=64, log=False, name=''):
    if log:
        scores = np.log(scores + 1e-8)

    if thresh is not None:
        if np.max(scores) < thresh:
            thresh = np.max(scores)
        scores[scores > thresh] = thresh
    bins = np.linspace(np.min(scores), np.max(scores), n_bins)
    scores_norm = scores[classes == 0]
    scores_ano = scores[classes == 1]

    plt.clf()
    plt.hist(scores_norm, bins, alpha=0.5, density=True, label='non-defects', color='cyan', edgecolor="black")
    plt.hist(scores_ano, bins, alpha=0.5, density=True, label='defects', color='crimson', edgecolor="black")

    ticks = np.linspace(np.min(scores), np.max(scores), 5)
    labels = ['{:.2f}'.format(i) for i in ticks[:-1]] + ['>' + '{:.2f}'.format(np.max(scores))]
    plt.xticks(ticks, labels=labels)
    plt.xlabel('Anomaly Score' if not log else 'Log Anomaly Score')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(axis='y')
    plt.show()


parser = argparse.ArgumentParser(description='ONNX Medical Images Anomaly Classification')
parser.add_argument('--model-path', type=str, required=True,
                    help='Path to ONNX classifier model file')
parser.add_argument('--dataset', type=str)
parser.add_argument('--batch-size', type=int, default=1)
parser.add_argument('--use_val', action='store_true',
                    help='Test on val data.')
parser.add_argument('--output-folder', type=str, default='output_folder',
                    help='Folder containing the .pkl reconstruction files')

parser.set_defaults(use_val=False)

args = parser.parse_args()

assert args.dataset in DATASETS

if __name__ == '__main__':
    onnx_session = load_onnx_model(args.model_path)

    # Data
    normal_paths = get_normal_images_paths()
    abnormal_paths = get_abnormal_images_paths()

    file_paths = normal_paths + abnormal_paths
    gt_labels = np.concatenate((np.zeros(len(normal_paths)), np.ones(len(abnormal_paths))))
    file_paths, gt_labels = shuffle(file_paths, gt_labels, random_state=12)  # only to visualize different labels
    
    get_auc(onnx_session, paths=file_paths, ground_truth_labels=gt_labels)
