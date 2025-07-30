import numpy as np
import nibabel as nib
import os
import tensorflow as tf
from tqdm.notebook import tqdm

import sys

from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy("mixed_float16")

repo_path = "/content/drive/MyDrive/Medical-Segmentation-Decathlon"
sys.path.append(repo_path)

def calculate_class_weights(label_files):
    """
    Calculates class weights based on voxel frequency in a list of NIfTI label files.
    """
    total_counts = {}
    total_voxels = 0

    print(f"Analyzing {len(label_files)} label files...")

    # First pass: Count the voxels for each class
    for file_path in tqdm(label_files):
        label_img = nib.load(file_path)
        label_data = label_img.get_fdata().astype(np.uint8)

        unique, counts = np.unique(label_data, return_counts=True)

        for cls, count in zip(unique, counts):
            total_counts[cls] = total_counts.get(cls, 0) + count
        total_voxels += label_data.size

    # Remove label 3 if it exists, as it's not used in BraTS
    if 3 in total_counts:
        del total_counts[3]

    sorted_counts = {k: total_counts[k] for k in sorted(total_counts.keys())}
    num_classes = len(sorted_counts)

    print("\n--- Voxel Counts ---")
    for cls, count in sorted_counts.items():
        percentage = (count / total_voxels) * 100
        print(f"Class {cls}: {count} voxels ({percentage:.4f}%)")

    # Second pass: Calculate weights using inverse frequency
    weights = {}
    for cls, count in sorted_counts.items():
        weight = total_voxels / (num_classes * count)
        weights[cls] = weight

    # Normalize weights so the smallest weight is 1.0
    min_weight = min(weights.values())
    normalized_weights = {cls: weight / min_weight for cls, weight in weights.items()}

    print("\n--- Calculated Weights ---")
    print("Inverse Frequency Weights:", weights)
    print("Normalized Weights (Recommended):", normalized_weights)

    print("\n--- Class Confirmation ---")
    unique_classes_found = sorted(total_counts.keys())
    print(f"Unique classes found in all files: {unique_classes_found}")
    print(f"Total number of unique classes: {len(unique_classes_found)}")

    return normalized_weights

# --- Main Execution (Corrected for your folder structure) ---

# --- Main Execution (Modified to filter hidden files) ---

# 1. Set the correct paths to your dataset folders
image_address = '/content/Task01_BrainTumour/imagesTr'
label_address = '/content/Task01_BrainTumour/labelsTr'

# 2. Instantiate the DataPipeline
pipeline = DataPipeline(image_address=image_address, label_address=label_address)

# 3. Get the list of ALL label file paths, ignoring hidden files
all_label_filenames = sorted([
    f for f in os.listdir(pipeline.label_address)
    # FIX: Add a check to ignore files starting with '._'
    if f.endswith(".nii.gz") and not f.startswith("._")
])

# Construct the full path by joining the directory path and the filename
all_label_paths = [os.path.join(pipeline.label_address, name) for name in all_label_filenames]


# 4. Run the calculation
if all_label_paths:
    calculated_weights = calculate_class_weights(all_label_paths)
else:
    print("No label files found. Please check your path.")