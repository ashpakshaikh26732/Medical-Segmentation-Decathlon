import tensorflow as tf
import yaml
import sys
import numpy as np
import nibabel as nib
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
import os

# --- Project Setup ---
repo_path = "/content/drive/MyDrive/Medical-Segmentation-Decathlon"
sys.path.append(repo_path)

from tensorflow.keras import mixed_precision
policy = tf.keras.mixed_precision.Policy("mixed_bfloat16")
mixed_precision.set_global_policy(policy)

from Project.src.models.swim_trans_unet import SwinTransUnet
from Project.src.models.transunet import TRANSUNET
from Project.src.models.unetpp import UNET_PLUS_PLUS

class Inference:
    """
    CORRECTED Inference engine matching training preprocessing exactly.
    
    Key fixes:
    1. Volume-level normalization (not patch-level)
    2. Explicit padding before patch extraction
    3. Fixed config key access for CT modality
    """
    def __init__(self, config_path: str, strategy: tf.distribute.Strategy) -> None:
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.strategy = strategy
        self.patch_shape = self.config['data']['image_patch_shape']
        self.inference_batch_size = self.config['data']['batch'] * strategy.num_replicas_in_sync

        model_registry = {
            "unet_plus_plus": UNET_PLUS_PLUS,
            'swin_transunet': SwinTransUnet,
            'transUnet': TRANSUNET
        }

        with self.strategy.scope():
            model_name = self.config['model']['name']
            if model_name == 'swin_transunet':
                model = model_registry[model_name](self.config)
            else:
                model = model_registry[model_name](self.config['data']['num_classes'])
            
            sample_input_shape = [1] + self.patch_shape
            _ = model(inputs=tf.zeros(sample_input_shape, dtype=tf.float32), training=False)
            print(f"Model '{model_name}' built successfully.")
        self._model = model
        self.optimizer = tf.keras.optimizers.AdamW(
            learning_rate=float(self.config['optimizer']['starting_lr']),
            weight_decay=float(self.config['optimizer']['weight_decay'])
        )
    
    def _initialize_optimizer_slots(self):
        strategy = tf.distribute.get_strategy()
        with strategy.scope():
            self.optimizer.build(self._model.trainable_variables)
            
    def _load_checkpoint(self) -> None:
        """Loads model weights from the checkpoint path in the config."""
        checkpoint_path = self.config['checkpoint']['checkpoint_path']
        with self.strategy.scope():
            self._initialize_optimizer_slots()
            checkpoint = tf.train.Checkpoint(
                model=self._model,
                optimizer=self.optimizer,
                epoch=tf.Variable(0, dtype=tf.int32)
            )
            status = checkpoint.restore(checkpoint_path).expect_partial()
            status.assert_nontrivial_match() 
            print(f"✅ Checkpoint loaded successfully from: {checkpoint_path}")

    def _resize_volume_like_training(self, volume_hwdc, target_shape_dhwc, method):
        """Resizes matching training pipeline exactly."""
        target_d, target_h, target_w = target_shape_dhwc[0], target_shape_dhwc[1], target_shape_dhwc[2]
        volume_dhwc = tf.transpose(volume_hwdc, perm=[2, 0, 1, 3])
        resized_hw = tf.image.resize(volume_dhwc, (target_h, target_w), method=method)
        transposed = tf.transpose(resized_hw, perm=[1, 2, 0, 3])
        resized_d = tf.image.resize(transposed, (target_w, target_d), method=method)
        final_volume = tf.transpose(resized_d, perm=[2, 0, 1, 3])
        final_volume.set_shape(target_shape_dhwc)
        return final_volume

    def _resize_back_to_original(self, volume_dhwc, original_spatial_shape_hwd, method):
        """Resizes back to original shape."""
        target_h, target_w, target_d = original_spatial_shape_hwd[0], original_spatial_shape_hwd[1], original_spatial_shape_hwd[2]
        resized_hw = tf.image.resize(volume_dhwc, (target_h, target_w), method=method)
        transposed = tf.transpose(resized_hw, perm=[1, 2, 0, 3])
        resized_d = tf.image.resize(transposed, (target_w, target_d), method=method)
        final_volume = resized_d
        final_volume.set_shape([target_h, target_w, target_d, volume_dhwc.shape[-1]])
        return final_volume

    def _normalize_volume_like_training(self, volume: tf.Tensor) -> tf.Tensor:
        """
        FIX #1: Normalize the ENTIRE VOLUME (not per-patch).
        This exactly matches pipeline.py's _val_normalization logic.
        """
        # Compute stats over spatial dims [0,1,2], keep channel dim
        min_value = tf.reduce_min(volume, axis=[0, 1, 2], keepdims=True)
        max_value = tf.reduce_max(volume, axis=[0, 1, 2], keepdims=True)
        
        modality = self.config['data']['modality']
        
        if modality == 'MRI':
            normalized = (volume - min_value) / (max_value - min_value + 1e-8)
        elif modality == 'CT':
            # FIX #3: Use correct config key structure
            clip_min = self.config['data']['modality']['CT']['clip_value_min']
            clip_max = self.config['data']['modality']['CT']['clip_value_max']
            clipped = tf.clip_by_value(volume, clip_min, clip_max)
            min_value = tf.reduce_min(clipped, axis=[0, 1, 2], keepdims=True)
            max_value = tf.reduce_max(clipped, axis=[0, 1, 2], keepdims=True)
            normalized = (clipped - min_value) / (max_value - min_value + 1e-8)
        else:
            normalized = (volume - min_value) / (max_value - min_value + 1e-8)

        # Z-score normalization
        mean = tf.math.reduce_mean(normalized, axis=[0, 1, 2], keepdims=True)
        std = tf.math.reduce_std(normalized, axis=[0, 1, 2], keepdims=True)
        return (normalized - mean) / (std + 1e-6)

    def _pad_volume_like_training(self, volume: tf.Tensor) -> tf.Tensor:
        """
        FIX #2: Add explicit padding matching training's _pad_volumes.
        Pads by half the patch size on all sides.
        """
        pad_d = self.patch_shape[0] // 2
        pad_h = self.patch_shape[1] // 2
        pad_w = self.patch_shape[2] // 2

        paddings = tf.constant([
            [pad_d, pad_d],
            [pad_h, pad_h],
            [pad_w, pad_w],
            [0, 0]  # No padding on channel dim
        ], dtype=tf.int32)

        return tf.pad(volume, paddings=paddings, mode='CONSTANT', constant_values=0)

    def _get_gaussian_mask(self):
        """Creates a Gaussian weighting mask for smooth blending."""
        patch_dims = self.patch_shape[:3]
        mask = np.zeros(patch_dims, dtype=np.float32)
        center_coords = [p // 2 for p in patch_dims]
        mask[tuple(center_coords)] = 1.0
        gaussian_mask = gaussian_filter(mask, sigma=[p / 8.0 for p in patch_dims])
        return tf.constant(gaussian_mask, dtype=tf.float32)

    def _run_inference(self, nifti_file_path: str):
        """Runs the corrected preprocessing, inference, and reconstruction pipeline."""
        
        # Load original volume
        data = nib.load(nifti_file_path)
        original_affine, original_spatial_shape, original_header = data.affine, data.shape[:3], data.header
        volume_np = data.get_fdata().astype(np.float32)
        if volume_np.ndim == 3:
            volume_np = np.expand_dims(volume_np, axis=-1)
        
        # Resize to target shape
        volume_tf_hwdc = tf.convert_to_tensor(volume_np, dtype=tf.float32)
        target_shape_dhwc = self.config['data']['image_shape']
        resized_volume_dhwc = self._resize_volume_like_training(
            volume_tf_hwdc, target_shape_dhwc, method='bilinear'
        )
        
        # FIX #1: Normalize the ENTIRE volume before patching
        print("Normalizing full volume (matching training)...")
        normalized_volume = self._normalize_volume_like_training(resized_volume_dhwc)
        
        # FIX #2: Pad the volume before patch extraction
        print("Padding volume (matching training)...")
        padded_volume = self._pad_volume_like_training(normalized_volume)
        
        # Now extract patches from the padded, normalized volume
        vol_d = tf.shape(padded_volume)[0]
        vol_h = tf.shape(padded_volume)[1]
        vol_w = tf.shape(padded_volume)[2]
        num_classes = self.config['data']['num_classes']
        num_channels = tf.shape(padded_volume)[-1]
        
        # Initialize accumulators (for the UNPADDED volume size)
        unpadded_d = tf.shape(normalized_volume)[0]
        unpadded_h = tf.shape(normalized_volume)[1]
        unpadded_w = tf.shape(normalized_volume)[2]
        
        prediction_accumulator = tf.zeros([unpadded_d, unpadded_h, unpadded_w, num_classes], dtype=tf.float32)
        count_accumulator = tf.zeros_like(prediction_accumulator)
        gaussian_mask = self._get_gaussian_mask()
        
        patch_d, patch_h, patch_w, _ = self.patch_shape
        stride_d, stride_h, stride_w = patch_d // 2, patch_h // 2, patch_w // 2

        # Extract patches (now from padded volume, so no need for SAME padding)
        patches = tf.extract_volume_patches(
            input=tf.expand_dims(padded_volume, 0), 
            ksizes=[1, patch_d, patch_h, patch_w, 1],
            strides=[1, stride_d, stride_h, stride_w, 1], 
            padding='VALID'  # Changed from SAME since we pre-padded
        )
        
        grid_dims = tf.shape(patches)[1:4]
        num_patches = tf.reduce_prod(grid_dims)
        patches_flat = tf.reshape(patches, [num_patches, patch_d, patch_h, patch_w, num_channels])
        
        # Distribute patches for inference
        patch_dataset = tf.data.Dataset.from_tensor_slices(patches_flat)
        distributed_dataset = self.strategy.experimental_distribute_dataset(
            patch_dataset.batch(self.inference_batch_size)
        )
        
        @tf.function
        def inference_step(batch_of_patches):
            # Patches are already normalized, just cast to float32 for model
            if self.config['model']['name'] == 'unet_plus_plus':
                logits = self._model(inputs=batch_of_patches, training=False)[-1]
            else:
                logits = self._model(inputs=batch_of_patches, training=False)
            return tf.cast(logits, tf.float32)
            
        all_predictions = []
        total_batches = int(np.ceil(num_patches / self.inference_batch_size))
        
        for batch in tqdm(distributed_dataset, desc="Inferring Patches", total=total_batches):
            per_replica_results = self.strategy.run(inference_step, args=(batch,))
            gathered_predictions = self.strategy.gather(per_replica_results, axis=0)
            all_predictions.append(gathered_predictions) 
            
        predictions_tensor = tf.concat(all_predictions, axis=0)
        
        # Reconstruct volume (accounting for padding offset)
        print("Reconstructing volume from patches...")
        pad_d = self.patch_shape[0] // 2
        pad_h = self.patch_shape[1] // 2
        pad_w = self.patch_shape[2] // 2
        
        patch_index = 0
        for z in range(grid_dims[0]):
            for y in range(grid_dims[1]):
                for x in range(grid_dims[2]):
                    if patch_index >= num_patches:
                        break
                    
                    pred_patch = predictions_tensor[patch_index]
                    patch_index += 1
                    
                    # Calculate position in PADDED space
                    z_start_padded = z * stride_d
                    y_start_padded = y * stride_h
                    x_start_padded = x * stride_w
                    
                    # Convert to UNPADDED space
                    z_start = z_start_padded - pad_d
                    y_start = y_start_padded - pad_h
                    x_start = x_start_padded - pad_w
                    
                    # Skip patches entirely outside unpadded volume
                    if (z_start >= unpadded_d or y_start >= unpadded_h or x_start >= unpadded_w or
                        z_start + patch_d < 0 or y_start + patch_h < 0 or x_start + patch_w < 0):
                        continue
                    
                    # Compute valid region overlap
                    z_start_valid = max(0, z_start)
                    y_start_valid = max(0, y_start)
                    x_start_valid = max(0, x_start)
                    
                    z_end_valid = min(unpadded_d, z_start + patch_d)
                    y_end_valid = min(unpadded_h, y_start + patch_h)
                    x_end_valid = min(unpadded_w, x_start + patch_w)
                    
                    # Extract corresponding slice from prediction
                    pred_z_start = z_start_valid - z_start
                    pred_y_start = y_start_valid - y_start
                    pred_x_start = x_start_valid - x_start
                    
                    pred_z_end = pred_z_start + (z_end_valid - z_start_valid)
                    pred_y_end = pred_y_start + (y_end_valid - y_start_valid)
                    pred_x_end = pred_x_start + (x_end_valid - x_start_valid)
                    
                    pred_slice = pred_patch[pred_z_start:pred_z_end, 
                                           pred_y_start:pred_y_end, 
                                           pred_x_start:pred_x_end, :]
                    mask_slice = gaussian_mask[pred_z_start:pred_z_end, 
                                               pred_y_start:pred_y_end, 
                                               pred_x_start:pred_x_end]
                    
                    weighted_pred = pred_slice * mask_slice[..., tf.newaxis]
                    weighted_count = tf.broadcast_to(mask_slice[..., tf.newaxis], tf.shape(weighted_pred))
                    
                    indices = tf.stack(tf.meshgrid(
                        tf.range(z_start_valid, z_end_valid),
                        tf.range(y_start_valid, y_end_valid),
                        tf.range(x_start_valid, x_end_valid),
                        indexing='ij'
                    ), axis=-1)
                    
                    prediction_accumulator = tf.tensor_scatter_nd_add(
                        prediction_accumulator, indices, weighted_pred
                    )
                    count_accumulator = tf.tensor_scatter_nd_add(
                        count_accumulator, indices, weighted_count
                    )

        final_logits_dhwc = prediction_accumulator / (count_accumulator + 1e-8)
        prediction_map_dhwc = tf.argmax(final_logits_dhwc, axis=-1)
        
        # Remap labels for Task 01
        if int(self.config['task']) == 1:
            print("Applying Task 01 label remap (3 -> 4)")
            three = tf.constant(3, dtype=prediction_map_dhwc.dtype)
            four = tf.constant(4, dtype=prediction_map_dhwc.dtype)
            prediction_map_dhwc = tf.where(tf.equal(prediction_map_dhwc, three), four, prediction_map_dhwc)
            
        prediction_map_dhwc = tf.expand_dims(tf.cast(prediction_map_dhwc, tf.float32), axis=-1)
        
        # Resize back to original dimensions
        final_map_hwdc = self._resize_back_to_original(
            prediction_map_dhwc, 
            original_spatial_shape, 
            method='nearest'
        )
        
        return tf.cast(tf.squeeze(final_map_hwdc), tf.uint8).numpy(), original_affine, volume_np, original_header

    def predict_from_file(self, nifti_file_path: str):
        """Main public method to run the complete inference pipeline."""
        print("1. Loading checkpoint...")
        self._load_checkpoint()
        
        print("\n2. Running corrected inference pipeline...")
        final_map, affine, original_img, header = self._run_inference(nifti_file_path)
        
        print("\n✅ Inference complete.")
        return final_map, affine, original_img, header