python -c "
from pathlib import Path
from convert_nifti_to_h5 import process_dataset
process_dataset(
    'brats20_data/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/', 
    'brats20_training_h5/', 
    'training'
)
"