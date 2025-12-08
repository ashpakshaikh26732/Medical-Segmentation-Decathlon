import tensorflow as tf
import yaml
import sys
import numpy as np
import nibabel as nib
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
import scipy.ndimage
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
    A production-grade, distributed inference orchestrator for 3D medical image segmentation.

    This engine implements a robust **Sliding Window Inference** strategy designed to 
    eliminate spatial alignment artifacts and patch boundary discontinuities. It ensures 
    pixel-perfect mapping between the model's output and the original anatomical space 
    by mirroring the training preprocessing pipeline exactly.

    Key Technical Features:
    -----------------------
    1. **Spatial Alignment Correction (The 'Ghost Padding' Fix):** Unlike standard inference loops that rely on `padding='SAME'`, this engine uses 
       **Explicit Symmetric Padding** combined with **Valid Extraction**. This guarantees 
       that coordinate systems remain deterministic, eliminating sub-voxel shifts caused 
       by asymmetric framework padding.

    2. **Gaussian Weighted Blending:** Overlapping patch predictions are aggregated using a Gaussian kernel. This 
       suppresses edge artifacts where model confidence is typically lower, resulting 
       in seamless volumetric reconstruction.

    3. **Isotropic Resampling & Normalization:** Handles on-the-fly resampling to target voxel spacing (e.g., 1mm isotropic) and 
       applies modality-specific normalization (Z-score for MRI, HU-clipping for CT) 
       to match the distribution seen during training.

    4. **Distributed Computation:** Leverages `tf.distribute.Strategy` to parallelize patch inference across available 
       accelerators (TPUs/GPUs), maximizing throughput for large 3D volumes.

    Attributes:
        config (dict): Configuration dictionary loaded from YAML.
        strategy (tf.distribute.Strategy): The active distribution strategy.
        patch_shape (list): Spatial dimensions of the input patches [D, H, W, C].
        inference_batch_size (int): Global batch size scaled by the number of replicas.
        _model (tf.keras.Model): The instantiated and restored deep learning model.
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
        checkpoint_path = '/kaggle/input/task2-transunet-checkpoints/ckpt-31'
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

    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 1: Load and Resample to Isotropic Spacing (MATCHES load_data())
    # ═══════════════════════════════════════════════════════════════════════════
    def _load_and_resample_to_isotropic(self, nifti_file_path: str):
        """
        Loads NIfTI and resamples to target spacing - EXACTLY matching training's load_data().
        
        This was MISSING in the previous inference code!
        """
        nii = nib.load(nifti_file_path)
        volume = nii.get_fdata().astype(np.float32)
        original_affine = nii.affine
        original_header = nii.header
        original_shape = volume.shape[:3]  # Original spatial shape before any processing
        original_volume = volume.copy()  # Keep original for return
        
        # Check if volume is already 4D (multi-channel like BrainTumour)
        has_channels = len(volume.shape) == 4
        
        if has_channels:
            num_channels = volume.shape[-1]
            spatial_volume = volume[..., 0]
        else:
            spatial_volume = volume
            num_channels = 1
        
        # Get spacing from header
        spacing = np.array(nii.header.get_zooms()[:3], dtype=np.float32)
        target_spacing = np.array(self.config['data']['target_spacing'], dtype=np.float32)
        resize_factor = spacing / target_spacing
        
        # Calculate new shape based on SPATIAL dimensions only
        new_shape = np.round(np.array(spatial_volume.shape) * resize_factor).astype(int)
        
        print(f"  Original spacing: {spacing}")
        print(f"  Target spacing: {target_spacing}")
        print(f"  Original shape: {spatial_volume.shape} → Resampled shape: {tuple(new_shape)}")
        
        # Resample with order=1 (bilinear) for images
        order = 1
        
        if has_channels:
            resampled_channels = []
            for c in range(num_channels):
                resampled = scipy.ndimage.zoom(
                    volume[..., c], 
                    zoom=new_shape / np.array(volume[..., c].shape), 
                    order=order
                )
                resampled_channels.append(resampled)
            resampled_volume = np.stack(resampled_channels, axis=-1)
        else:
            resampled_volume = scipy.ndimage.zoom(
                volume, 
                zoom=new_shape / np.array(volume.shape), 
                order=order
            )
            resampled_volume = np.expand_dims(resampled_volume, axis=-1)
        
        return resampled_volume, original_affine, original_header, original_shape, spacing, original_volume

    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 2: Resize Volume (MATCHES Generator.resize_volume())
    # ═══════════════════════════════════════════════════════════════════════════
    def _resize_volume_like_training(self, volume_dhwc: tf.Tensor) -> tf.Tensor:
        """
        Resizes volume to target shape - EXACTLY matching training's resize_volume().
        
        Input: (D, H, W, C) - already in correct format after resampling
        Output: (D, H, W, C) at target_image_shape
        """
        target_image_shape = self.config['data']['image_shape']
        
        # 1. Resize Height and Width: input shape (D, H, W, C)
        resized_hw = tf.image.resize(
            volume_dhwc,
            (target_image_shape[1], target_image_shape[2]),  # (H, W)
            method='bilinear'
        )
        
        # 2. Transpose for depth resizing: new shape (H, W, D, C)
        transposed = tf.transpose(resized_hw, perm=[1, 2, 0, 3])
        
        # 3. Resize Depth: input shape (H, W, D, C), resizing the (W, D) plane
        resized_d = tf.image.resize(
            transposed,
            (target_image_shape[2], target_image_shape[0]),  # (W, D)
            method='bilinear'
        )
        
        # 4. Transpose back to original format: final shape (D, H, W, C)
        final_volume = tf.transpose(resized_d, perm=[2, 0, 1, 3])
        final_volume.set_shape(target_image_shape)
        
        return final_volume

    def _resize_back_to_original(self, volume_dhwc: tf.Tensor, original_shape_dhw, method='nearest') -> tf.Tensor:
        """
        Resizes prediction back to original spatial shape.
        """
        target_d, target_h, target_w = original_shape_dhw
        
        # 1. Resize H, W
        resized_hw = tf.image.resize(volume_dhwc, (target_h, target_w), method=method)
        
        # 2. Transpose to (H, W, D, C)
        transposed = tf.transpose(resized_hw, perm=[1, 2, 0, 3])
        
        # 3. Resize W, D
        resized_d = tf.image.resize(transposed, (target_w, target_d), method=method)
        
        # 4. Transpose back to (D, H, W, C)
        final_volume = tf.transpose(resized_d, perm=[2, 0, 1, 3])
        
        return final_volume

    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 3: Normalization (MATCHES DataPipeline.normalization())
    # ═══════════════════════════════════════════════════════════════════════════
    def _normalize_volume_like_training(self, volume: tf.Tensor) -> tf.Tensor:
        """
        Applies nnU-Net style normalization - EXACTLY matching training's normalization().
        
        MRI: Per-channel foreground-only Z-score
        CT: HU clipping + global Z-score
        """
        dtype = volume.dtype
        image = tf.cast(volume, tf.float32)
        
        # Handle NaN values
        image = tf.where(tf.math.is_nan(image), tf.zeros_like(image), image)
        
        modality = self.config['data']['modality']
        
        if modality == 'MRI':
            # MRI: Per-channel foreground-only Z-score normalization
            def normalize_channel(channel):
                """Z-score normalize using foreground voxels only."""
                fg_mask = tf.cast(tf.not_equal(channel, 0), tf.float32)
                num_fg = tf.reduce_sum(fg_mask) + 1e-8
                
                fg_mean = tf.reduce_sum(channel * fg_mask) / num_fg
                fg_var = tf.reduce_sum(((channel - fg_mean) ** 2) * fg_mask) / num_fg
                fg_std = tf.sqrt(fg_var + 1e-8)
                
                normalized = (channel - fg_mean) / fg_std
                return normalized
            
            channels = tf.unstack(image, axis=-1)
            normalized_channels = [normalize_channel(ch) for ch in channels]
            image = tf.stack(normalized_channels, axis=-1)
            
        elif modality == 'CT':
            # CT: HU clipping + global Z-score normalization
            clip_min = float(self.config['data']['CT_clip_value_min'])
            clip_max = float(self.config['data']['CT_clip_value_max'])
            
            image = tf.clip_by_value(image, clip_min, clip_max)
            
            mean = tf.reduce_mean(image)
            std = tf.math.reduce_std(image)
            image = (image - mean) / (std + 1e-8)
        
        return tf.cast(image, dtype)

    # ═══════════════════════════════════════════════════════════════════════════
    # Gaussian Mask for Smooth Blending
    # ═══════════════════════════════════════════════════════════════════════════
    def _get_gaussian_mask(self):
        """Creates a Gaussian weighting mask for smooth patch blending."""
        patch_dims = self.patch_shape[:3]
        mask = np.zeros(patch_dims, dtype=np.float32)
        center_coords = [p // 2 for p in patch_dims]
        mask[tuple(center_coords)] = 1.0
        gaussian_mask = gaussian_filter(mask, sigma=[p / 8.0 for p in patch_dims])
        return tf.constant(gaussian_mask, dtype=tf.float32)

    # ═══════════════════════════════════════════════════════════════════════════
    # Main Inference Pipeline (NO PADDING - direct sliding window)
    # ═══════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════
    #  STEP 4: Extract patches (CORRECTED: Explicit Padding + VALID Extraction)
    # ═══════════════════════════════════════════════════════════════════════════
    def _extract_patches_robust(self, volume, patch_shape, strides):
        """
        Manually pads volume to be divisible by stride/patch size, 
        then uses 'VALID' extraction to avoid hidden TF padding shifts.
        """
        vol_d, vol_h, vol_w, _ = volume.shape
        pd, ph, pw, _ = patch_shape
        sd, sh, sw = strides

        # 1. Calculate required padding to ensure the last patch fits perfectly
        # Formula: We want (Dimension + Pad) to be covered by patches
        
        # Calculate how much padding is needed for 'VALID' to cover everything
        # We need: (Input + Pad - Patch) % Stride == 0
        
        pad_d = max(0, pd - sd) # Minimum overlap padding
        pad_h = max(0, ph - sh)
        pad_w = max(0, pw - sw)

        # Extend padding to cover the full volume if strides leave a remainder
        # We perform symmetric padding to center the brain
        out_d = int(np.ceil((vol_d - pd) / sd) * sd) + pd
        out_h = int(np.ceil((vol_h - ph) / sh) * sh) + ph
        out_w = int(np.ceil((vol_w - pw) / sw) * sw) + pw

        pad_d_total = max(0, out_d - vol_d)
        pad_h_total = max(0, out_h - vol_h)
        pad_w_total = max(0, out_w - vol_w)

        # Split padding (before, after)
        p_d_bef = pad_d_total // 2
        p_d_aft = pad_d_total - p_d_bef
        
        p_h_bef = pad_h_total // 2
        p_h_aft = pad_h_total - p_h_bef
        
        p_w_bef = pad_w_total // 2
        p_w_aft = pad_w_total - p_w_bef

        paddings = tf.constant([
            [p_d_bef, p_d_aft],
            [p_h_bef, p_h_aft],
            [p_w_bef, p_w_aft],
            [0, 0] # Channel dim
        ])

        # 2. Pad the volume
        volume_padded = tf.pad(volume, paddings, mode='CONSTANT', constant_values=0)
        
        # 3. Extract with VALID (No hidden shifts!)
        patches = tf.extract_volume_patches(
            input=tf.expand_dims(volume_padded, 0),
            ksizes=[1, pd, ph, pw, 1],
            strides=[1, sd, sh, sw, 1],
            padding='VALID'
        )
        
        # Return pads so we can crop them out later
        return patches, volume_padded.shape, (p_d_bef, p_h_bef, p_w_bef)

    # ═══════════════════════════════════════════════════════════════════════════
    #  Main Inference Pipeline (FIXED)
    # ═══════════════════════════════════════════════════════════════════════════
    def _run_inference(self, nifti_file_path: str):
        # ... [Steps 1-3 remain exactly the same: Load, Resize, Normalize] ...
        print("Step 1: Loading and resampling to isotropic spacing...")
        resampled_volume, original_affine, original_header, original_shape, original_spacing, original_volume = \
            self._load_and_resample_to_isotropic(nifti_file_path)
        
        print("Step 2: Resizing to target image_shape...")
        volume_tf = tf.convert_to_tensor(resampled_volume, dtype=tf.float32)
        resized_volume = self._resize_volume_like_training(volume_tf)
        
        print("Step 3: Normalizing volume (nnU-Net style)...")
        normalized_volume = self._normalize_volume_like_training(resized_volume)

        # ─────────────────────────────────────────────────────────────────────
        # STEP 4: Extract patches (ROBUST METHOD)
        # ─────────────────────────────────────────────────────────────────────
        print("Step 4: Extracting patches (Explicit Padding + VALID)...")
        num_classes = self.config['data']['num_classes']
        num_channels = normalized_volume.shape[-1]
        
        patch_d, patch_h, patch_w, _ = self.patch_shape
        stride_d, stride_h, stride_w = patch_d // 2, patch_h // 2, patch_w // 2 # 50% overlap

        # Call the new robust extraction
        patches, padded_shape, pads_before = self._extract_patches_robust(
            normalized_volume, 
            self.patch_shape, 
            (stride_d, stride_h, stride_w)
        )
        
        grid_dims = patches.shape[1:4] # The D, H, W grid sizes
        num_patches = int(np.prod(grid_dims))
        patches_flat = tf.reshape(patches, [num_patches, patch_d, patch_h, patch_w, num_channels])
        print(f"  Total patches: {num_patches} (grid: {grid_dims})")
        
        # ─────────────────────────────────────────────────────────────────────
        # STEP 5: Run model inference (Same as before)
        # ─────────────────────────────────────────────────────────────────────
        print("Step 5: Running model inference...")
        patch_dataset = tf.data.Dataset.from_tensor_slices(patches_flat)
        distributed_dataset = self.strategy.experimental_distribute_dataset(
            patch_dataset.batch(self.inference_batch_size)
        )
        
        @tf.function
        def inference_step(batch_of_patches):
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
        
        # ─────────────────────────────────────────────────────────────────────
        # STEP 6: Reconstruct volume (INTO PADDED SHAPE)
        # ─────────────────────────────────────────────────────────────────────
        print("Step 6: Reconstructing volume...")
        
        # Use the PADDED shape for the accumulator
        pad_vol_d, pad_vol_h, pad_vol_w, _ = padded_shape
        
        prediction_accumulator = tf.zeros([pad_vol_d, pad_vol_h, pad_vol_w, num_classes], dtype=tf.float32)
        count_accumulator = tf.zeros_like(prediction_accumulator)
        gaussian_mask = self._get_gaussian_mask()
        
        patch_index = 0
        
        # Iterate over the grid determined by VALID extraction
        for z in range(grid_dims[0]):
            for y in range(grid_dims[1]):
                for x in range(grid_dims[2]):
                    pred_patch = predictions_tensor[patch_index]
                    patch_index += 1
                    
                    # Coordinate in the PADDED volume
                    z_start = z * stride_d
                    y_start = y * stride_h
                    x_start = x * stride_w
                    
                    z_end = z_start + patch_d
                    y_end = y_start + patch_h
                    x_end = x_start + patch_w
                    
                    # Create indices
                    indices = tf.stack(tf.meshgrid(
                        tf.range(z_start, z_end),
                        tf.range(y_start, y_end),
                        tf.range(x_start, x_end),
                        indexing='ij'
                    ), axis=-1)
                    
                    # Accumulate (using Gaussian blending)
                    prediction_accumulator = tf.tensor_scatter_nd_add(
                        prediction_accumulator, indices, pred_patch * gaussian_mask[..., tf.newaxis]
                    )
                    count_accumulator = tf.tensor_scatter_nd_add(
                        count_accumulator, indices, tf.broadcast_to(gaussian_mask[..., tf.newaxis], tf.shape(pred_patch))
                    )

        final_logits_padded = prediction_accumulator / (count_accumulator + 1e-8)
        
        # ─────────────────────────────────────────────────────────────────────
        # STEP 6.5: CROP BACK TO ORIGINAL SIZE (The Critical Fix)
        # ─────────────────────────────────────────────────────────────────────
        print("Step 6.5: Cropping padding to restore alignment...")
        p_d, p_h, p_w = pads_before
        
        # Original resized shape
        orig_h, orig_w, orig_d = normalized_volume.shape[:3] # Note: normalized_volume is (D,H,W,C)
        
        # Slice out the center
        final_logits_dhwc = final_logits_padded[
            p_d : p_d + normalized_volume.shape[0],
            p_h : p_h + normalized_volume.shape[1],
            p_w : p_w + normalized_volume.shape[2],
            :
        ]
        
        prediction_map_dhwc = tf.argmax(final_logits_dhwc, axis=-1)
        
        # ─────────────────────────────────────────────────────────────────────
        # STEP 7: Remap labels (Task 01 fix)
        # ─────────────────────────────────────────────────────────────────────
        if int(self.config['task']) == 1:
            three = tf.constant(3, dtype=prediction_map_dhwc.dtype)
            four = tf.constant(4, dtype=prediction_map_dhwc.dtype)
            prediction_map_dhwc = tf.where(tf.equal(prediction_map_dhwc, three), four, prediction_map_dhwc)
            
        prediction_map_dhwc = tf.expand_dims(tf.cast(prediction_map_dhwc, tf.float32), axis=-1)
        
        # ─────────────────────────────────────────────────────────────────────
        # STEP 8: Resize back
        # ─────────────────────────────────────────────────────────────────────
        # [Same as your original code]
        resampled_shape = resampled_volume.shape[:3]
        resized_to_resampled = self._resize_back_to_original(prediction_map_dhwc, resampled_shape, method='nearest')
        final_map = self._resize_back_to_original(resized_to_resampled, original_shape, method='nearest')
        
        return tf.cast(tf.squeeze(final_map), tf.uint8).numpy(), original_affine, original_volume, original_header

    def predict_from_file(self, nifti_file_path: str):
        """Main public method to run the complete inference pipeline."""
        print("=" * 70)
        print("CORRECTED INFERENCE PIPELINE (No unnecessary padding)")
        print("=" * 70)
        
        print("\n1. Loading checkpoint...")
        self._load_checkpoint()
        
        print("\n2. Running inference pipeline...")
        final_map, affine, original_img, header = self._run_inference(nifti_file_path)
        
        print("\n" + "=" * 70)
        print("✅ Inference complete.")
        print("=" * 70)
        
        return final_map, affine, original_img, header
    
    def save_prediction(self, prediction: np.ndarray, affine: np.ndarray, 
                       header, output_path: str):
        """Saves the prediction as a NIfTI file."""
        nii = nib.Nifti1Image(prediction.astype(np.uint8), affine, header)
        nib.save(nii, output_path)
        print(f"Saved prediction to: {output_path}")