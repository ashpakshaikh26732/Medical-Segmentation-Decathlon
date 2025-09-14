import numpy as np
import nibabel as nib
import os
import tensorflow as tf
from tqdm.notebook import tqdm
import sys


repo_path = "/content/drive/MyDrive/Medical-Segmentation-Decathlon"
sys.path.append(repo_path)


def calculate_class_weights(label_files):
    """
    Calculates class weights based on voxel frequency in a list of NIfTI label files.
    This version correctly handles all 4 classes in the BraTS dataset.
    """
    total_counts = {}
    total_voxels = 0

    print(f"Analyzing {len(label_files)} label files...")


    for file_path in tqdm(label_files):
        label_img = nib.load(file_path)
        label_data = label_img.get_fdata().astype(np.uint8)

        unique, counts = np.unique(label_data, return_counts=True)

        for cls, count in zip(unique, counts):
            total_counts[cls] = total_counts.get(cls, 0) + count
        total_voxels += label_data.size



    sorted_counts = {k: total_counts[k] for k in sorted(total_counts.keys())}
    num_classes = len(sorted_counts)

    print("\n--- Voxel Counts ---")
    for cls, count in sorted_counts.items():
        percentage = (count / total_voxels) * 100
        print(f"Class {cls}: {count} voxels ({percentage:.4f}%)")

 
    weights = {}
    for cls, count in sorted_counts.items():

        if count == 0:
            weights[cls] = 0
        else:
            weight = total_voxels / (num_classes * count)
            weights[cls] = weight


    non_zero_weights = [w for w in weights.values() if w > 0]
    if not non_zero_weights:

        return {cls: 1.0 for cls in sorted_counts.keys()}

    min_weight = min(non_zero_weights)
    normalized_weights = {cls: weight / min_weight for cls, weight in weights.items()}

    print("\n--- Calculated Weights ---")
    print("Inverse Frequency Weights:", weights)
    print("Normalized Weights (Recommended):", normalized_weights)

    print("\n--- Class Confirmation ---")
    unique_classes_found = sorted(total_counts.keys())
    print(f"Unique classes found in all files: {unique_classes_found}")
    print(f"Total number of unique classes: {len(unique_classes_found)}")

    return normalized_weights


label_address = '/kaggle/working/Task01_BrainTumour/labelsTr'


all_label_filenames = sorted([
    f for f in os.listdir(label_address)
    if f.endswith(".nii.gz") and not f.startswith("._")
])


all_label_paths = [os.path.join(label_address, name) for name in all_label_filenames]


if all_label_paths:
    calculated_weights = calculate_class_weights(all_label_paths)
    print("\nFinal list of weights to use in your YAML file:")

    weight_list = [calculated_weights.get(i, 0.0) for i in sorted(calculated_weights.keys())]
    print(weight_list)
else:
    print("No label files found. Please check your path.")