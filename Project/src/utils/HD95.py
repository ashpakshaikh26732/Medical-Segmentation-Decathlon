import tensorflow as tf
import numpy as np
import scipy.ndimage
from scipy.spatial.distance import cdist



def erosion_function(value):
    """Performs erosion and casts the output to int32."""
    eroded_mask = scipy.ndimage.binary_erosion(value)
    return eroded_mask.astype(np.int32)

def compute_surface_distances(true_coords, pred_coords):
    """
    Calculates the surface distances between two sets of 3D coordinates efficiently.
    Note: Expects NumPy arrays of shape (N, 3) for coordinates.
    """
    if true_coords.shape[0] == 0 or pred_coords.shape[0] == 0:
        return np.array([]), np.array([])

    distance_matrix = cdist(true_coords, pred_coords)
    dists_true_to_pred = np.min(distance_matrix, axis=1)
    dists_pred_to_true = np.min(distance_matrix, axis=0)
    return dists_true_to_pred, dists_pred_to_true


y_true_batch = tf.random.uniform(shape=(27, 80, 80, 52, 1), minval=0, maxval=3, dtype=tf.int32)
y_pred_batch = tf.random.uniform(shape=(27, 80, 80, 52, 3), minval=0, maxval=2)


y_pred_batch_argmax = tf.argmax(y_pred_batch, axis=-1, output_type=tf.int32)



num_patches_in_batch = y_true_batch.shape[0]
for patch_idx in range(num_patches_in_batch):
    print(f"==========================================")
    print(f"      Processing Patch #{patch_idx + 1} / {num_patches_in_batch}      ")
    print(f"==========================================")


    y_true_patch = tf.squeeze(y_true_batch[patch_idx], axis=-1)
    y_pred_patch = y_pred_batch_argmax[patch_idx]


    for class_idx in range(3):
        print(f"--- Calculating distances for Class {class_idx} ---")


        binary_mask_true = tf.cast(tf.where(y_true_patch == class_idx, 1, 0), dtype=tf.int32)
        binary_mask_pred = tf.cast(tf.where(y_pred_patch == class_idx, 1, 0), dtype=tf.int32)


        eroded_true = tf.numpy_function(erosion_function, [binary_mask_true], tf.int32)
        eroded_pred = tf.numpy_function(erosion_function, [binary_mask_pred], tf.int32)
        surface_mask_true = binary_mask_true - eroded_true
        surface_mask_pred = binary_mask_pred - eroded_pred


        coords_true = tf.where(surface_mask_true == 1).numpy()
        coords_pred = tf.where(surface_mask_pred == 1).numpy()


        dists_t2p, dists_p2t = compute_surface_distances(coords_true, coords_pred)

        if dists_t2p.size == 0 or dists_p2t.size == 0:
            print("No surface points found for one of the masks. Skipping.\n")
            continue

        hd95 = max(np.percentile(dists_t2p, 95), np.percentile(dists_p2t, 95))
        assd = (np.mean(dists_t2p) + np.mean(dists_p2t)) / 2.0

        print(f"HD95: {hd95:.4f}, ASSD: {assd:.4f}\n")