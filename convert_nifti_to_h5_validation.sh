python -c "
from pathlib import Path
from convert_nifti_to_h5 import process_dataset
process_dataset(
    'brats20_data/BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData/', 
    'brats20_validation_h5/', 
    'validation'
)
"