
#### This is the official repo of "Masked Autoencoders for Unsupervised Anomaly Detection in Medical Images"@KES2023.

### Acknowledgement: 
This code is mostly built on: [MAE](https://github.com/facebookresearch/mae). We thank 🙏 the authors for sharing their code.

### 📜 Arxiv Link: https://arxiv.org/pdf/2307.07534.pdf

### 🌟 Overview

We tackle anomaly detection in medical images training our framework using only healthy samples. We propose to use the Masked Autoencoder model to learn
the structure of the normal samples, then train an anomaly classifier on top of the difference between the original image and the reconstruction provided by the masked autoencoder. We train the anomaly classifier in a supervised manner using as negative
samples the reconstruction of the healthy scans, while as positive samples, we use pseudo-abnormal scans obtained via our novel
pseudo-abnormal module. The pseudo-abnormal module alters the reconstruction of the normal samples by changing the intensity of several regions.

### ⚙️ Environment Setup

An interactive setup script is provided to create a conda environment with the correct dependencies:

```bash
source ./setup_env.sh
```

The script auto-detects your CUDA version, installs PyTorch, and runs verification tests. Alternatively, install manually:

```bash
conda create -n mae-medical python=3.8 -y
conda activate mae-medical
# Install PyTorch for your CUDA version (see https://pytorch.org)
pip install -r requirements.txt
```

After installation, verify everything works:

```bash
python test_installation.py
```

### Download the data sets.

1.1 BraTS2020: 
   Download the BraTS2020 dataset from [Kaggle](https://www.kaggle.com/datasets/awsaf49/brats2020-training-data).

1.2 Luna16: Download the Luna16 data set from [Luna16GrandChallenge](https://luna16.grand-challenge.org/Data/).

2  The splits used in this work can be found in the ```dataset``` folder.

2.1 For BraTS2020, we released the name of each slice.

2.2 For Luna16, we released the row number (from candidates.csv) of each region.

#### Data Preparation (BraTS2020 H5 → NPY)

If your BraTS2020 data is in H5 format, convert it to NPY and organize into train/val/test splits:

```bash
python convert_h5_to_npy.py
```

This reads H5 files from `dataset/BraTS2020_training_data/content/data/`, converts each `image` array to `.npy`, and places them into `dataset/BraTS2020_training_data/split/{train,val,test}/{normal,abnormal}/` according to the split text files. The script supports resuming — already-converted files are skipped automatically.


### 💻 Running the code.
#### Step 1: Pretraining phase.

```bash
python3 main_pretrain.py \
    --batch_size 64 \
    --model mae_vit_base_patch16 \
    --mask_ratio 0.75 \
    --epochs 1600 \
    --warmup_epochs 40 \
    --output_dir mae_mask_ratio_0.75 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --batch_size=128 \
    --data_path path_to_the_normal_samples
```

#### Step 2: Extract reconstructions.
```bash
python3 extract_reconstructions.py \
    --dataset=brats --mask-ratio=0.85 \
    --model-path=path_to_the_saved_model/checkpoint-1599.pth \
    --batch-size=64 --num-trials=4 \
    --output-folder=output_folder
```

Notice that you have to set the paths to the data set in the ```extract_reconstructions.py``` file and run the above command for the train, val and test splits. The script supports resuming — already-generated `.pkl` files are skipped.

#### Step 3: Train the anomaly classifier.
```bash
python3 main_finetune.py \
    --batch_size 128 \
    --model vit_base_patch16 \
    --finetune path_to_the_saved_model.75_brats/checkpoint-1599.pth \
    --epochs 100 \
    --weight_decay 0.05 --drop_path 0.1 \
    --nb_classes 2 \
    --aa=None \
    --output_dir output_folder \
    --data_path path_to_the_reconstructions_obtained_in_the_previous_step
```

#### Step 4: Evaluate the model.
```bash
python3 evaluate_sup.py --dataset=brats \
    --model-path=path_to_the_best_model.pth --batch-size=64
```

#### End-to-End Evaluation (Quick Start)

To run the full pipeline on test data with the provided pretrained models in one go:

```bash
bash e2e_evaluate.sh
```

This sequentially converts H5→NPY, extracts reconstructions, and evaluates AUROC.

### 📦 ONNX Export

Both the MAE reconstruction model and the anomaly classifier can be exported to ONNX for deployment.

#### Export MAE (random masking)
```bash
python export_onnx_mae.py \
    --model-path models/brats_pretrained.pth \
    --output-path exported_onnx_models/mae_brats_random.onnx \
    --dataset brats --opset-version 14
```

#### Export MAE (deterministic / grid masking)
```bash
python export_deterministic_mae.py \
    --model-path models/brats_pretrained.pth \
    --output-path exported_onnx_models/mae_brats_deterministic.onnx \
    --dataset brats --opset-version 14
```

#### Export Anomaly Classifier
```bash
python export_onnx_classifier.py \
    --model-path models/brats_finetuned.pth \
    --output-path exported_onnx_models/classifier_brats.onnx \
    --dataset brats --arch vit_base_patch16 \
    --opset-version 14 --verify
```

Convenience wrappers are also available: `export_onnx_mae.sh` and `export_onnx_anomaly_classifier.sh`.

#### ONNX Graph Surgery

The `graph_surgery/` directory contains post-export graph simplification scripts for hardware-accelerator compatibility:

- **`graph_surgery_mae_grid_masking.py`** — Replaces `GatherElements` with simpler `Gather` ops and rewrites the 6D Reshape+Einsum unpatchify pattern into Reshape+Transpose sequences.
- **`graph_surgery_anomaly_classifier_split_reducemean.py`** — Splits a large `ReduceMean` along the feature dimension into smaller chunks to stay within hardware dimension limits.

#### ONNX Runtime Inference

Drop-in ONNX Runtime replacements exist for the PyTorch extraction and evaluation scripts:

```bash
# Extract reconstructions with ONNX Runtime
python3 extract_reconstructions_onnx.py \
    --dataset=brats --mask-ratio=0.75 \
    --model-path=exported_onnx_models/mae_brats_deterministic_grid_masking_simplified.onnx \
    --batch-size=1 --num-trials=4 \
    --output-folder=output_folder_onnx --test

# Evaluate with ONNX Runtime
python3 evaluate_sup_onnx.py \
    --dataset=brats \
    --model-path=exported_onnx_models/classifier_brats_split.onnx \
    --batch-size=1 --output-folder=output_folder_onnx
```

### 🚀 Results and trained models.


<table>
<tr>
    <td>Dataset</td> 
    <td>Pretrained Model</td>
    <td>Finetuned Model</td>  
    <td>AUROC</td>  
</tr>
  
<tr>
    <td>BraTS2020</td> 
    <td><a href="https://drive.google.com/file/d/1QxFHy8nYeaj5OPQExmcbf9PQNzMOhoCy/view?usp=sharing">GDrive</a></td>
    <td><a href="https://drive.google.com/file/d/1x7gSu3G2Cd4n_Gl8yDmpy7wzOW8XTN5J/view?usp=drive_link">GDrive</a></td>
    <td>0.899</td>
</tr>

<tr>
    <td>LUNA16</td> 
    <td><a href="https://drive.google.com/file/d/1ALMc7s8_WozNm1rckSo1gEgpB4GNpAJs/view?usp=sharing">GDrive</a></td>
    <td><a href="https://drive.google.com/file/d/1Yc_dQ6Gb5tn6GM7BDvmL9-YGY9GIHxW9/view?usp=sharing">GDrive</a></td>
     <td>0.634</td>
</tr>
 
 


</table>

### 📁 Repository Structure

| File / Directory | Description |
|---|---|
| `main_pretrain.py` | MAE pretraining on healthy samples |
| `extract_reconstructions.py` | Generate reconstructions for train/val/test splits |
| `main_finetune.py` | Train the anomaly classifier on reconstruction errors |
| `evaluate_sup.py` | Compute AUROC on test data (PyTorch) |
| `convert_h5_to_npy.py` | BraTS2020 H5 → NPY conversion with split organization |
| `setup_env.sh` | Interactive conda environment setup |
| `test_installation.py` | Verify all dependencies and model creation |
| `e2e_evaluate.sh` | End-to-end evaluation pipeline script |
| `export_onnx_mae.py` | Export MAE to ONNX (random or deterministic masking) |
| `export_deterministic_mae.py` | Export MAE to ONNX with fixed grid masking |
| `export_onnx_classifier.py` | Export anomaly classifier to ONNX |
| `extract_reconstructions_onnx.py` | ONNX Runtime reconstruction extraction |
| `evaluate_sup_onnx.py` | ONNX Runtime evaluation |
| `graph_surgery/` | ONNX graph simplification for hardware accelerators |
| `models/` | Pretrained & finetuned model weights |
| `exported_onnx_models/` | Exported ONNX model files |
| `dataset/split/` | Train/val/test split definitions for BraTS2020 and LUNA16 |

### 🔒 License
The present code is released under the Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) license.


### Citation
```
@inproceedings{Georgescu-KES-2023,
  title="{Masked Autoencoders for Unsupervised Anomaly Detection in Medical Images}",
  author={Georgescu, Mariana-Iuliana},
  booktitle={Proceedings of KES},
  year={2023}
}
```
