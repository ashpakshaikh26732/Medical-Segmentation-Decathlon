import sys
import tensorflow as tf
import nibabel as nib
import os
import numpy as np
import tensorflow_graphics.geometry.transformation as tfg_transformation
import keras_cv


repo_path = "/content/drive/MyDrive/Medical-Segmentation-Decathlon"
sys.path.append(repo_path)

class OHEM_Genrator :
    """
    A Python generator class for loading and performing initial preparation of 3D
    NIfTI medical imaging data on-the-fly.

    This class is designed to be the first stage of a high-performance data pipeline,
    handling the "out-of-graph" file I/O operations before passing data to TensorFlow.
    It reads .nii.gz files, splits them into training and validation sets, and yields
    them as NumPy arrays.
    """
    def __init__(self, config , image_address , label_address , hard_patch_map = None):
        """
        Initializes the NiftiGenerator.

        Args:
            config (dict): The main configuration dictionary for the experiment.
            image_address (str): The path to the directory containing image files.
            label_address (str): The path to the directory containing label files.
        """
        self.config = config
        self.image_address = image_address
        self.label_address = label_address
        self.val_count = config['data']['val_count']
        self.hard_patch_map= hard_patch_map

    def getFullImageAddress(self, image) :
        """Constructs the full path for an image filename."""
        return os.path.join(self.image_address,image)

    def getFullLabelAddress(self, label):
        """Constructs the full path for a label filename."""
        return os.path.join(self.label_address, label )

    def convert_to_tensor(self, image , label , hard_coords_for_volume_ragged):
        """Converts NumPy arrays to TensorFlow tensors."""
        image = tf.convert_to_tensor(image)
        label = tf.convert_to_tensor(label)
        return image , label , hard_coords_for_volume_ragged

    def resize_volume(self, image, label ,hard_coords_for_volume_ragged ):
        """
        Resizes a 3D volume to a target shape specified in the config using
        the modern tf.image.resize function for robust interpolation.
        """

        target_image_shape = self.config['data']['image_shape']
        target_label_shape = self.config['data']['label_shape']


        resized_hw = tf.image.resize(
            image,
            (target_image_shape[1], target_image_shape[2]),
            method='bilinear'
        )

        transposed = tf.transpose(resized_hw, perm=[1, 2, 0, 3])

        resized_d = tf.image.resize(
            transposed,
            (target_image_shape[2], target_image_shape[0]), 
            method='bilinear'
        )

        final_image = tf.transpose(resized_d, perm=[2, 0, 1, 3])
        final_image.set_shape(target_image_shape)



        resized_hw_label = tf.image.resize(
            label,
            (target_label_shape[1], target_label_shape[2]),
            method='nearest'
        )
        transposed_label = tf.transpose(resized_hw_label, perm=[1, 2, 0, 3])
        resized_d_label = tf.image.resize(
            transposed_label,
            (target_label_shape[2], target_label_shape[0]),
            method='nearest'
        )
        final_label = tf.transpose(resized_d_label, perm=[2, 0, 1, 3])
        final_label.set_shape(target_label_shape)

        return final_image, final_label , hard_coords_for_volume_ragged

    def cast(self, image , label , hard_coords_for_volume_ragged , dtype = tf.float32):
        """Casts image and label tensors to a specified data type."""
        image = tf.cast(image , dtype=dtype)
        label = tf.cast(label , dtype = dtype)
        return image , label ,hard_coords_for_volume_ragged

    def load_data(self, path: str) -> np.ndarray:
        """
        Loads a single NIfTI file from a given path.

        Args:
            path (str): The full path to the .nii.gz file.

        Returns:
            np.ndarray: The image or label data as a 4D NumPy array (H, W, D, C).
        """
        data = nib.load(path)
        data = data.get_fdata().astype(np.float32)
        if len(data.shape) == 3:
            data = np.expand_dims(data, axis=-1)

        return data

    def get_data(self):
        """
        Scans the data directories to get sorted lists of image and label filenames.

        Returns:
            Tuple[List[str], List[str]]: A tuple containing the list of image
            filenames and the list of label filenames.
        """
        images = sorted([
            f for f in os.listdir(self.image_address)
            if f.endswith(".nii.gz") and not f.startswith("._")
        ])

        labels = sorted([
            f for f in os.listdir(self.label_address)
            if f.endswith(".nii.gz") and not f.startswith("._")
        ])
        return images , labels

    def train_data_genrator(self):
        """
        A Python generator that yields training samples.

        It loads image and label volumes from the training split of the dataset
        on-the-fly.

        Yields:
            Tuple[np.ndarray, np.ndarray]: A tuple containing the 4D image volume
            and the 4D label volume as NumPy arrays.
        """
        images  ,labels = self.get_data()

        train_images_paths = images[self.val_count:]
        train_labels_paths = labels[self.val_count:]
        train_images_paths = list(map(self.getFullImageAddress, train_images_paths))
        train_labels_paths = list(map(self.getFullLabelAddress, train_labels_paths))

        for img_path , label_path in zip(train_images_paths, train_labels_paths):
            image_volume = self.load_data(img_path)
            label_volume = self.load_data(label_path)

            hard_coords = self.hard_patch_map.get(img_path , [])

            if not hard_coords:
                # Create an empty tensor for the values, but with the correct inner shape
                values = tf.zeros(shape=(0, 3), dtype=tf.int32)
                # The row_splits for an empty rank-2 tensor is just [0]
                row_splits = tf.constant([0], dtype=tf.int64)
                ragged_coords = tf.RaggedTensor.from_tensor(
                    tf.zeros(shape=(0, 3), dtype=tf.int32)
                )
            else:
                ragged_coords = tf.ragged.constant(hard_coords, dtype=tf.int32)            

            yield image_volume , label_volume , ragged_coords

    def val_data_genrator(self):
        """
        A Python generator that yields validation samples.

        It loads image and label volumes from the validation split of the dataset
        on-the-fly.

        Yields:
            Tuple[np.ndarray, np.ndarray]: A tuple containing the 4D image volume
            and the 4D label volume as NumPy arrays.
        """
        images  ,labels = self.get_data()

        val_images_paths = images[:self.val_count]
        val_labels_paths = labels[:self.val_count]

        val_images_paths = list(map(self.getFullImageAddress, val_images_paths))
        val_labels_paths = list(map(self.getFullLabelAddress, val_labels_paths))

        for img_path , label_path  in zip(val_images_paths , val_labels_paths):

            val_images_volume = self.load_data(img_path)
            val_labels_volume = self.load_data(label_path)

            hard_coords = self.hard_patch_map.get(img_path , [])

            if not hard_coords:
                # Create an empty tensor for the values, but with the correct inner shape
                values = tf.zeros(shape=(0, 3), dtype=tf.int32)
                # The row_splits for an empty rank-2 tensor is just [0]
                row_splits = tf.constant([0], dtype=tf.int64)
                ragged_coords = tf.RaggedTensor.from_tensor(
                    tf.zeros(shape=(0, 3), dtype=tf.int32)
                )
            else:
                ragged_coords = tf.ragged.constant(hard_coords, dtype=tf.int32)                   

            yield val_images_volume , val_labels_volume ,ragged_coords




class OHEM_PatchSampler :
    def __init__(self , config):

        self.stage_patches_per_volume = config['data']['stage_patches_per_volume']


        self.intra_class_weights = list(config['data']['stage2_intra_class_ratio'])
        self.fg_ratio = config['data']['stage3_fg_ratio']
        
        self.stage2_intra_class_weights = list(config['data']['stage2_intra_class_ratio'])        

        self.stage3_hard_sample_ratio = config['data']['stage3_hard_sample_ratio']
        self.class_names = list(config['data']['class_names'])

        self.image_patch = list(config['data']['image_patch_shape'])

        self.label_patch = list(config['data']['label_patch_shape'])
        # self.hard_patch_map = None 
        # self.start_table = None
        # self.count_table = None
    
    def _extract_patches(self, volume, centers, patch_shape):
        """
        Extracts 3D patches from a volume at specified center coordinates using
        the SOTA nnU-Net "intelligent slicing" approach.

        This method is a fully graph-native, high-performance implementation that
        avoids the CPU bottleneck of tf.gather_nd by using tf.map_fn to apply
        a robust tf.slice operation in parallel for each patch.

        Args:
            volume (tf.Tensor): The 4D volume to extract from (D, H, W, C).
            centers (tf.Tensor): A tensor of center coordinates of shape [num_patches, 3].
            patch_shape (list or tuple): The spatial shape of the patches, e.g., (80, 80, 52, 4).

        Returns:
            tf.Tensor: A tensor of extracted patches with shape
                       [num_patches, patch_d, patch_h, patch_w, channels].
        """

        patch_shape_tf = tf.convert_to_tensor(patch_shape[:3], dtype=tf.int32)

        def _slice_one_patch(center):

            corner_spatial = center - patch_shape_tf // 2


            corner_spatial = tf.maximum(corner_spatial, 0)


            begin_coord = tf.concat([corner_spatial, [0]], axis=0)


            num_channels = tf.shape(volume)[-1]
            size_coord = tf.concat([patch_shape_tf, [num_channels]], axis=0)


            patch = tf.slice(volume, begin=begin_coord, size=size_coord)
            return patch


        output_signature = tf.TensorSpec(shape=patch_shape, dtype=volume.dtype)

        all_patches = tf.map_fn(
            fn=_slice_one_patch,
            elems=centers,
            fn_output_signature=output_signature
        )
        return all_patches

    # def set_hard_patchs(self, hard_patch_metadata_list):
    #     """
    #     Pre-processes hard patch metadata using the 'Two Tables' method.
    #     """
    #     print("🧠 Pre-processing hard patch metadata for Stage 3...")

    #     # This also needs to be added to your __init__
    #     if not hasattr(self, 'mega_coords_tensor'):
    #         self.mega_coords_tensor = None

    #     if not hard_patch_metadata_list:
    #         print("⚠️ Warning: No hard patch metadata provided. Stage 3 will use fallback sampling.")
    #         self.mega_coords_tensor = tf.zeros(shape=(0, 3), dtype=tf.int32)
    #         # Initialize two empty tables
    #         self.start_table = tf.lookup.StaticHashTable(
    #             tf.lookup.KeyValueTensorInitializer(tf.constant([], dtype=tf.string), tf.constant([], dtype=tf.int32)),
    #             default_value=0
    #         )
    #         self.count_table = tf.lookup.StaticHashTable(
    #             tf.lookup.KeyValueTensorInitializer(tf.constant([], dtype=tf.string), tf.constant([], dtype=tf.int32)),
    #             default_value=0
    #         )
    #         return

    #     # 1. Group coordinates by volume_id
    #     temp_map = {}
    #     for volume_id, coords in hard_patch_metadata_list:
    #         if volume_id not in temp_map:
    #             temp_map[volume_id] = []
    #         temp_map[volume_id].append(coords)

    #     volume_ids = list(temp_map.keys())

    #     # 2. Flatten coordinates and create separate lists for starts and counts
    #     all_coords_list = []
    #     start_indices = []
    #     counts = []
    #     current_index = 0
    #     for vid in volume_ids:
    #         coords = temp_map[vid]
    #         num_coords = len(coords)

    #         start_indices.append(current_index)
    #         counts.append(num_coords)

    #         all_coords_list.extend(coords)
    #         current_index += num_coords

    #     # 3. Create the final Tensors
    #     self.mega_coords_tensor = tf.constant(all_coords_list, dtype=tf.int32)
    #     starts_tensor = tf.constant(start_indices, dtype=tf.int32)
    #     counts_tensor = tf.constant(counts, dtype=tf.int32)

    #     # 4. Create two separate tables, each mapping keys to scalar values
    #     self.start_table = tf.lookup.StaticHashTable(
    #         tf.lookup.KeyValueTensorInitializer(keys=volume_ids, values=starts_tensor),
    #         default_value=0 # If key not found, start index is 0
    #     )
    #     self.count_table = tf.lookup.StaticHashTable(
    #         tf.lookup.KeyValueTensorInitializer(keys=volume_ids, values=counts_tensor),
    #         default_value=0 # If key not found, count is 0
    #     )
    #     print(f"✅ Pre-processing complete. Flattened {len(all_coords_list)} total hard patches.")




    def stage3_honring(self, image, label , hard_coords_for_volume_ragged  ):
        """
        Implements Stage 2: Refinement with Probabilistic Rare-Class Oversampling.

        This method robustly handles all edge cases, including empty class pools,
        and uses TensorFlow-native loops (`tf.range`) to ensure it can be compiled
        into a high-performance graph.
        """
        hard_coords_for_volume = hard_coords_for_volume_ragged.to_tensor() 

        num_fg_patches = tf.cast(self.stage_patches_per_volume * self.fg_ratio, dtype=tf.int32)
        num_hard_examples = tf.cast(self.stage_patches_per_volume * self.stage3_hard_sample_ratio , dtype = tf.int32)
        num_random_patches = tf.cast(self.stage_patches_per_volume - num_fg_patches - num_hard_examples, dtype=tf.int32)


        class_coords_list = []
      
        for i in range(1, len(self.class_names)):
            coords = tf.where(label == i)
            coords = tf.cast(coords[:, :3], dtype=tf.int32)
            class_coords_list.append(coords)


        class_coords_ragged = tf.ragged.stack(class_coords_list)
        all_fg_indices = tf.concat(class_coords_list, axis=0)
        
        def sample_with_foreground():

            
            class_logits = tf.math.log([self.stage2_intra_class_weights])

            chosen_class_indices = tf.random.categorical(logits=class_logits, num_samples=num_fg_patches)[0] 
            
            foreground_centers_ta = tf.TensorArray(dtype=tf.int32, size=num_fg_patches, dynamic_size=False)

            for i in tf.range(num_fg_patches):
                class_id_to_sample = chosen_class_indices[i]
                
                pool_of_coords_slice = class_coords_ragged[tf.cast(class_id_to_sample, tf.int32)]
                pool_of_coords = tf.reshape(pool_of_coords_slice, [-1, 3])
                num_coords_in_pool = tf.shape(pool_of_coords)[0]

                def pick_from_this_pool():

                    random_index = tf.random.uniform(shape=(), maxval=num_coords_in_pool, dtype=tf.int32)
                    return pool_of_coords[random_index]
                
                def pick_from_any_pool():

                    num_total_fg_coords = tf.shape(all_fg_indices)[0]
                    random_coord_index = tf.random.uniform(shape=(), maxval=num_total_fg_coords, dtype=tf.int32)
                    return all_fg_indices[random_coord_index]

                coord = tf.cond(num_coords_in_pool > 0,
                                true_fn=pick_from_this_pool,
                                false_fn=pick_from_any_pool)
                
                foreground_centers_ta = foreground_centers_ta.write(i, coord)
            
            return foreground_centers_ta.stack()

        def sample_with_none():

            return tf.zeros(shape=(0, 3), dtype=tf.int32)


        foreground_centers = tf.cond(tf.shape(all_fg_indices)[0] == 0,
                                    true_fn=sample_with_none,
                                    false_fn=sample_with_foreground)
                                    

        num_fg_sampled = tf.shape(foreground_centers)[0]
        # code for stage 3 hard 

        num_available_hard = tf.shape(hard_coords_for_volume)[0]


        def got_more_hard_examples():
            """Case 1: We have more hard examples than we need. Sample without replacement."""
            return tf.random.shuffle(hard_coords_for_volume)[:num_hard_examples]

        def got_some_hard_examples():
            """Case 2: We have some, but not enough. Sample with replacement."""
            random_indices = tf.random.uniform(
                shape=[num_hard_examples],
                maxval=num_available_hard,
                dtype=tf.int32
            )
            return tf.gather(hard_coords_for_volume, random_indices)

        def sample_with_no_hard_examples():
            """Case 3: Fallback. No hard examples for this volume."""

            return tf.zeros(shape=(0, 3), dtype=tf.int32)


        hard_centers = tf.cond(
            num_available_hard >= num_hard_examples,
            true_fn=got_more_hard_examples,
            false_fn=lambda: tf.cond(
                num_available_hard > 0,
                true_fn=got_some_hard_examples,
                false_fn=sample_with_no_hard_examples
            )
        )
        num_hard_examples_sampled = tf.shape(hard_centers)[0]
        # end code for stage 3 hard 
        num_total_random_patches = num_random_patches + (num_fg_patches - num_fg_sampled) + (num_hard_examples - num_hard_examples_sampled)

        volume_shape = tf.shape(image)
        max_d, max_h, max_w = volume_shape[0] - self.image_patch[0], volume_shape[1] - self.image_patch[1], volume_shape[2]-self.image_patch[2]


        random_d_coords = tf.random.uniform(shape=[num_total_random_patches], minval=0, maxval=max_d, dtype=tf.int32)
        random_h_coords = tf.random.uniform(shape=[num_total_random_patches], minval=0, maxval=max_h, dtype=tf.int32)
        random_w_coords = tf.random.uniform(shape=[num_total_random_patches], minval=0, maxval=max_w, dtype=tf.int32)

        random_centers = tf.stack([random_d_coords, random_h_coords, random_w_coords], axis=1)
        
        foreground_centers.set_shape([None, 3])
        random_centers.set_shape([None, 3])
        
        all_centers = tf.concat([foreground_centers, random_centers , hard_centers], axis=0)
    
        image_patchs = self._extract_patches( image , all_centers , self.image_patch )
        label_patchs = self._extract_patches( label , all_centers , self.label_patch )
        
        return image_patchs , label_patchs 


class RandomElasticDeformation3D(tf.keras.layers.Layer):
    """
    A high-performance 3D elastic deformation layer optimized for TPUs.
    
    This implementation leverages bfloat16 precision to halve memory bandwidth
    requirements and uses a fully vectorized trilinear interpolation for maximum speed.
    """
    def __init__(self,
                 grid_size=(4, 4, 4),
                 alpha=35.0,
                 sigma=2.5,
                 **kwargs):
        super().__init__(**kwargs)
        self.grid_size = grid_size
        self.alpha = tf.constant(alpha, dtype=tf.bfloat16)
        self.sigma = tf.constant(sigma, dtype=tf.bfloat16)

    def _separable_gaussian_filter_3d(self, tensor, sigma):
        """Applies a fast, separable 3D Gaussian filter."""
        kernel_size = tf.cast(2 * tf.round(3 * sigma) + 1, dtype=tf.int32)
        ax = tf.range(-tf.cast(kernel_size // 2, tf.bfloat16) + 1.0, 
                      tf.cast(kernel_size // 2, tf.bfloat16) + 1.0)
        kernel_1d = tf.exp(-(ax**2) / (2.0 * sigma**2))
        kernel_1d = kernel_1d / tf.reduce_sum(kernel_1d)
        
        filter_d = tf.cast(tf.reshape(kernel_1d, [-1, 1, 1, 1, 1]), dtype=tensor.dtype)
        filter_h = tf.cast(tf.reshape(kernel_1d, [1, -1, 1, 1, 1]), dtype=tensor.dtype)
        filter_w = tf.cast(tf.reshape(kernel_1d, [1, 1, -1, 1, 1]), dtype=tensor.dtype)

        tensor = tf.nn.convolution(
            tensor, filter_d, strides=[1, 1, 1, 1, 1], padding='SAME'
        )
        tensor = tf.nn.convolution(
            tensor, filter_h, strides=[1, 1, 1, 1, 1], padding='SAME'
        )
        tensor = tf.nn.convolution(
            tensor, filter_w, strides=[1, 1, 1, 1, 1], padding='SAME'
        )
        return tensor

    def call(self, image_volume, label_volume):
        original_image_dtype = image_volume.dtype
        image_volume = tf.cast(image_volume, dtype=tf.bfloat16)
        
        was_batched = True
        if image_volume.shape.rank == 4:
            was_batched = False
            image_volume = tf.expand_dims(image_volume, axis=0)
            label_volume = tf.expand_dims(label_volume, axis=0)

        input_shape = tf.shape(image_volume)
        B, D, H, W = input_shape[0], input_shape[1], input_shape[2], input_shape[3]
        
        coarse_flow = tf.random.uniform(
            shape=(B, self.grid_size[0], self.grid_size[1], self.grid_size[2], 3),
            minval=-1, maxval=1, dtype=tf.bfloat16)

        # Reshape the 5D tensor to 4D to use tf.image.resize, then reshape back.
        flow = tf.reshape(coarse_flow, [B * self.grid_size[0], self.grid_size[1], self.grid_size[2], 3])
        flow = tf.image.resize(flow, size=[H, W], method='bicubic')
        flow = tf.reshape(flow, [B, self.grid_size[0], H, W, 3])
        
        flow = tf.transpose(flow, perm=[0, 2, 3, 1, 4])
        flow = tf.reshape(flow, [B * H * W, self.grid_size[0], 3])
        flow = tf.image.resize(tf.expand_dims(flow, axis=1), size=[1, D], method='bicubic')
        flow = tf.squeeze(flow, axis=1)
        flow = tf.reshape(flow, [B, H, W, D, 3])
        flow = tf.transpose(flow, perm=[0, 3, 1, 2, 4])

        # --- THE FIX IS HERE ---
        # Recast flow to bfloat16, as tf.image.resize upcasts bicubic output to float32.
        flow = tf.cast(flow, dtype=tf.bfloat16)

        flow_components = tf.unstack(flow, axis=-1)
        smoothed_components = []
        for component in flow_components:
            smoothed_component = self._separable_gaussian_filter_3d(
                component[..., tf.newaxis], self.sigma
            )
            smoothed_components.append(smoothed_component[..., 0])
        smoothed_flow = tf.stack(smoothed_components, axis=-1)
        
        flow = smoothed_flow * self.alpha

        grid_d, grid_h, grid_w = tf.meshgrid(
            tf.range(D, dtype=tf.bfloat16),
            tf.range(H, dtype=tf.bfloat16),
            tf.range(W, dtype=tf.bfloat16),
            indexing='ij'
        )
        grid = tf.stack([grid_d, grid_h, grid_w], axis=-1)
        warp_grid = tf.expand_dims(grid, 0) + flow

        warp_grid_floor = tf.floor(warp_grid)
        t = warp_grid - warp_grid_floor
        
        d0 = tf.cast(warp_grid_floor[..., 0], tf.int32)
        h0 = tf.cast(warp_grid_floor[..., 1], tf.int32)
        w0 = tf.cast(warp_grid_floor[..., 2], tf.int32)

        d1 = tf.clip_by_value(d0 + 1, 0, D - 1)
        h1 = tf.clip_by_value(h0 + 1, 0, H - 1)
        w1 = tf.clip_by_value(w0 + 1, 0, W - 1)
        d0 = tf.clip_by_value(d0, 0, D - 1)
        h0 = tf.clip_by_value(h0, 0, H - 1)
        w0 = tf.clip_by_value(w0, 0, W - 1)

        c000 = tf.gather_nd(image_volume, tf.stack([d0, h0, w0], axis=-1), batch_dims=1)
        c001 = tf.gather_nd(image_volume, tf.stack([d0, h0, w1], axis=-1), batch_dims=1)
        c010 = tf.gather_nd(image_volume, tf.stack([d0, h1, w0], axis=-1), batch_dims=1)
        c011 = tf.gather_nd(image_volume, tf.stack([d0, h1, w1], axis=-1), batch_dims=1)
        c100 = tf.gather_nd(image_volume, tf.stack([d1, h0, w0], axis=-1), batch_dims=1)
        c101 = tf.gather_nd(image_volume, tf.stack([d1, h0, w1], axis=-1), batch_dims=1)
        c110 = tf.gather_nd(image_volume, tf.stack([d1, h1, w0], axis=-1), batch_dims=1)
        c111 = tf.gather_nd(image_volume, tf.stack([d1, h1, w1], axis=-1), batch_dims=1)

        td, th, tw = t[..., 0:1], t[..., 1:2], t[..., 2:3]
        c00 = c000 * (1 - tw) + c001 * tw
        c01 = c010 * (1 - tw) + c011 * tw
        c10 = c100 * (1 - tw) + c101 * tw
        c11 = c110 * (1 - tw) + c111 * tw
        c0 = c00 * (1 - th) + c01 * th
        c1 = c10 * (1 - th) + c11 * th
        deformed_image = c0 * (1 - td) + c1 * td
        
        deformed_image = tf.cast(deformed_image, original_image_dtype)

        nearest_indices_float = tf.round(warp_grid)
        nearest_d = tf.clip_by_value(tf.cast(nearest_indices_float[..., 0], tf.int32), 0, D - 1)
        nearest_h = tf.clip_by_value(tf.cast(nearest_indices_float[..., 1], tf.int32), 0, H - 1)
        nearest_w = tf.clip_by_value(tf.cast(nearest_indices_float[..., 2], tf.int32), 0, W - 1)
        deformed_label = tf.gather_nd(label_volume, tf.stack([nearest_d, nearest_h, nearest_w], axis=-1), batch_dims=1)
        
        if not was_batched:
            deformed_image = tf.squeeze(deformed_image, axis=0)
            deformed_label = tf.squeeze(deformed_label, axis=0)
            
        return deformed_image, deformed_label

class OHEM_DataPipeline : 
    def __init__(self, config , image_address , label_address , hard_patchs_list= None):
        """
        Initializes the main DataPipeline orchestrator.

        This class acts as a factory for creating state-of-the-art tf.data.Dataset
        objects. It composes the Generator and PatchSampler classes to build a
        complete, end-to-end, high-performance data pipeline.

        Args:
            config (dict): The main configuration dictionary for the experiment.
            image_address (str): The path to the directory containing image files.
            label_address (str): The path to the directory containing label files.
        """

        hard_patch_map= {}

        if hard_patchs_list : 
            print("🧠 Pre-processing hard patch metadata for the generator...")
            for volume_id_bytes , coords in hard_patchs_list : 
                volume_id = volume_id_bytes.decode('utf-8')
                if volume_id not in hard_patch_map : 
                    hard_patch_map[volume_id] = [] 
                hard_patch_map[volume_id].append(coords.tolist())
            

        num_classes = config['data']['num_classes']
        self.config = config 
        self.genrator = OHEM_Genrator(
            config ,
            image_address=image_address ,
            label_address=label_address,
            hard_patch_map = hard_patch_map
            )
        self.patch_sampler = OHEM_PatchSampler(config)
        self.rotator = keras_cv.layers.RandomRotation(
            factor = 0.15 , 
            interpolation="bilinear", 
            segmentation_classes=num_classes
            )
        self.zoomer = keras_cv.layers.RandomZoom(
            height_factor=0.2, 
            interpolation="bilinear",
                  
            )
        self.randomElasticDeformation3D = RandomElasticDeformation3D() 
        self.patch_shape = list(config['data']['image_patch_shape']) 
        self.final_batch_size = config['data']['batch'] * config['data']['num_replicas']
    
    def _geometric_augmentations(self, image, label , hard_coords_for_volume_ragged ):
        """
        Applies a sequence of random, in-graph 3D geometric augmentations.

        This method implements the state-of-the-art augmentation strategy
        popularized by frameworks like nnU-Net. All operations are native
        TensorFlow or KerasCV, ensuring a high-performance, graph-native pipeline.
        The augmentations include random mirroring, 3D rotation, 3D zoom, and
        3D elastic deformation.

        Args:
            image (tf.Tensor): The 4D input image volume (D, H, W, C).
            label (tf.Tensor): The 4D input label volume (D, H, W, C).

        Returns:
            Tuple[tf.Tensor, tf.Tensor]: A tuple containing the augmented image
            and label volumes.
        """

        # fliping 
        along_depth = tf.random.uniform(())>0.5
        along_height = tf.random.uniform(())>0.5
        along_width = tf.random.uniform(())>0.5

        if along_depth : 
            image = tf.reverse(image , axis =[0] )
            label = tf.reverse(label, axis = [0]) 
        if along_height : 
            image = tf.reverse(image , axis =[1] )
            label = tf.reverse(label , axis = [1]) 
        if along_width : 
            image = tf.reverse(image  , axis = [2]) 
            label = tf.reverse(label , axis =[2])
        
        # rotation 
        # along xy
        if tf.random.uniform(())>0.5: 


            augmented = self.rotator({
                    'images': image , 
                    'segmentation_masks': label
                })
            aug_image ,aug_label = augmented['images'], augmented['segmentation_masks']
        
            image = tf.cast(aug_image, image.dtype)
            label = tf.cast(aug_label, label.dtype)
        
        # along xz
        if tf.random.uniform(())>0.5: 

            image = tf.transpose(image, perm=[1, 0, 2, 3])
            label = tf.transpose(label, perm=[1, 0, 2, 3])
            augmented = self.rotator({
                    'images': image , 
                    'segmentation_masks': label
                })
            aug_image ,aug_label = augmented['images'], augmented['segmentation_masks']
            image = tf.cast(aug_image, image.dtype)
            label = tf.cast(aug_label, label.dtype)

            image = tf.transpose(image, perm=[1, 0, 2, 3])
            label = tf.transpose(label, perm=[1, 0, 2, 3])

 

        # along yz
        if tf.random.uniform(())>0.5: 


            image = tf.transpose(image, perm=[2, 1, 0, 3])
            label = tf.transpose(label, perm=[2, 1, 0, 3])
            augmented = self.rotator({
                'images': image, 
                'segmentation_masks': label
            })
            aug_image, aug_label = augmented['images'], augmented['segmentation_masks']

            image = tf.cast(aug_image, image.dtype)
            label = tf.cast(aug_label, label.dtype)

            image = tf.transpose(image, perm=[2, 1, 0, 3])
            label = tf.transpose(label, perm=[2, 1, 0, 3])

        # zoom 
        # along xy plan 

        if tf.random.uniform(())>0.5: 

            augmented = self.zoomer({
                    'images': image , 
                    'segmentation_masks': label
                })
            aug_image ,aug_label = augmented['images'], augmented['segmentation_masks']
            image = tf.cast(aug_image, image.dtype)
            label = tf.cast(aug_label, label.dtype)
        # along xz
        if tf.random.uniform(())>0.5: 
            image = tf.transpose(image, perm=[1, 0, 2, 3])
            label = tf.transpose(label, perm=[1, 0, 2, 3])
            augmented = self.zoomer({
                    'images': image , 
                    'segmentation_masks': label
                })
            aug_image ,aug_label = augmented['images'], augmented['segmentation_masks']

            image = tf.cast(aug_image, image.dtype)
            label = tf.cast(aug_label, label.dtype)

            image = tf.transpose(image, perm=[1, 0, 2, 3])
            label = tf.transpose(label, perm=[1, 0, 2, 3])


        # along yz
        if tf.random.uniform(())>0.5: 

            image = tf.transpose(image, perm=[2, 1, 0, 3])
            label = tf.transpose(label, perm=[2, 1, 0, 3])
            augmented = self.zoomer({
                'images': image, 
                'segmentation_masks': label
            })
            aug_image, aug_label = augmented['images'], augmented['segmentation_masks']

            image = tf.cast(aug_image, image.dtype)
            label = tf.cast(aug_label, label.dtype)

            image = tf.transpose(image, perm=[2, 1, 0, 3])
            label = tf.transpose(label, perm=[2, 1, 0, 3])

        
        # elastic_deformation 
        image , label = self.randomElasticDeformation3D(image , label)

        return image , label , hard_coords_for_volume_ragged

    def _intensity_augmentations(self , image,label ):
        """
        Applies a sequence of random, in-graph 3D intensity augmentations to a
        single patch.

        This method follows the SOTA nnU-Net augmentation suite, making the model
        robust to variations in scanner brightness, contrast, and noise. All
        operations are native TensorFlow, ensuring a high-performance pipeline.
        The label is passed through unmodified.

        Args:
            image (tf.Tensor): The 4D input image patch (D, H, W, C).
            label (tf.Tensor): The 4D input label patch (D, H, W, C).

        Returns:
            Tuple[tf.Tensor, tf.Tensor]: A tuple containing the augmented image
            patch and the original, unmodified label patch.
        """
        min_value = tf.reduce_min(image , axis =[0,1,2] , keepdims=True)
        max_value  =tf.reduce_max(image , axis = [0,1,2] , keepdims=True)
        if self.config['data']['modality']=='MRI':
            min_max_normalized_image = (image - min_value)/(max_value-min_value + 1e-8)

        elif (self.config['data']['modality']) =='CT':
            min_max_normalized_image = tf.clip_by_value(image , clip_value_min=self.config['data']['modality']['CT']['clip_value_min'] ,clip_value_max=self.config['data']['modality']['CT']['clip_value_max'])
            min_value = tf.reduce_min(min_max_normalized_image , axis = [0,1,2] , keepdims=True)
            max_value = tf.reduce_max(min_max_normalized_image ,axis =[0,1,2] , keepdims=True)
            min_max_normalized_image = (min_max_normalized_image-min_value)/(max_value-min_value+1e-8)

        image = tf.image.random_brightness(min_max_normalized_image , max_delta=0.1)
        image = tf.image.random_contrast(image, lower=0.7 , upper=1.1)
        Gaussian_noise = tf.random.normal((tf.shape(image)),stddev=0.1 , dtype = image.dtype)
        image = image + Gaussian_noise
        gama_value = tf.random.uniform(() , minval=0.7 , maxval=1.5 ,  dtype=image.dtype)
        image = tf.clip_by_value(image, 0.0, 1.0)
        image = tf.math.pow(image , gama_value)
        image = tf.clip_by_value(image ,clip_value_min=0.0 , clip_value_max=1.0 )

        mean = tf.math.reduce_mean(image ,axis=[0, 1, 2], keepdims=True)
        std  = tf.math.reduce_std(image , axis=[0, 1, 2], keepdims=True)
        image  = (image-mean)/(std + 1e-6)

        return image , label

    def _pad_volumes(self, image_volume, label_volume , hard_coords_for_volume_ragged ):
        """
        Applies pre-emptive padding to 3D volumes to prevent out-of-bounds
        errors during patch extraction.

        This method adds padding to all spatial dimensions of the image and label
        volumes. The amount of padding is half the patch size, ensuring that any
        patch centered on a voxel from the original volume will be fully contained
        within the new, padded volume. This is a key SOTA technique for building a
        robust data pipeline.

        Args:
            image_volume (tf.Tensor): The 4D input image volume (D, H, W, C).
            label_volume (tf.Tensor): The 4D input label volume (D, H, W, C).

        Returns:
            Tuple[tf.Tensor, tf.Tensor]: A tuple containing the padded image and
            label volumes.
        """

        pad_d = self.patch_shape[0] // 2
        pad_h = self.patch_shape[1] // 2
        pad_w = self.patch_shape[2] // 2

        paddings = [
            [pad_d, pad_d],
            [pad_h, pad_h],
            [pad_w, pad_w],
            [0, 0]
        ]
        paddings_tensor = tf.constant(paddings, dtype=tf.int32)

        padded_image = tf.pad(
            image_volume,
            paddings=paddings_tensor,
            mode='CONSTANT',
            constant_values=0
        )
        padded_label = tf.pad(
            label_volume,
            paddings=paddings_tensor,
            mode='CONSTANT',
            constant_values=0
        )

        return padded_image, padded_label , hard_coords_for_volume_ragged
    
    def _val_normalization(self, image , label) : 
        """
        Applies a standardized normalization to the validation and test data.

        This ensures that the model sees data in the same distribution as the
        training data, but without any stochastic augmentations.

        Args:
            image (tf.Tensor): The 4D input image patch (D, H, W, C).
            label (tf.Tensor): The 4D input label patch (D, H, W, C).

        Returns:
            Tuple[tf.Tensor, tf.Tensor]: A tuple containing the normalized
            image patch and the original label patch.
        """
        min_value = tf.reduce_min(image , axis = [0,1,2] , keepdims=True )
        max_value  =tf.reduce_max(image , axis =[0,1,2] ,  keepdims=True)
        if self.config['data']['modality']=='MRI':
            min_max_normalized_image = (image - min_value)/(max_value-min_value + 1e-8)

        elif (self.config['data']['modality']) =='CT':
            min_max_normalized_image = tf.clip_by_value(image , clip_value_min=self.config['data']['modality']['CT']['clip_value_min'] ,clip_value_max=self.config['data']['modality']['CT']['clip_value_max'])
            min_value = tf.reduce_min(min_max_normalized_image , axis =[0,1,2] , keepdims=True)
            max_value = tf.reduce_max(min_max_normalized_image , axis =  [0,1,2] , keepdims=True)
            min_max_normalized_image = (min_max_normalized_image-min_value)/(max_value-min_value+1e-8)


        mean = tf.math.reduce_mean(min_max_normalized_image ,axis=[0, 1, 2], keepdims=True)
        std  = tf.math.reduce_std(min_max_normalized_image , axis=[0, 1, 2], keepdims=True)
        image  = (min_max_normalized_image-mean)/(std + 1e-6)

        return image , label 

    def _remap_labels(self, image, label):
        """
        Maps the BraTS labels {0, 1, 2, 4} to sequential labels {0, 1, 2, 3}.
        This version is corrected for dtype consistency.
        """

        three = tf.constant(3, dtype=label.dtype)

        label = tf.where(tf.equal(label, 4), three, label)
        return image, label


    def get_dataset(self, is_training = True ): 
        """
        Builds and returns a complete, state-of-the-art tf.data.Dataset.

        This is the main public method of the orchestrator. It assembles all the
        SOTA components (Generator, Sampler, Augmentations) into a single,
        high-performance pipeline, which can be dynamically configured by the
        main training loop.

        Args:
            stage (str): The current training stage, which controls the sampling
                         strategy (e.g., 'stage1_foundational').
            hard_patchs_list (list, optional): A list of hard patch coordinates,
                                           used only for Stage 3. Defaults to None.
            is_training (bool): If True, the training pipeline with full
                                augmentations and shuffling is built. Otherwise,
                                a simpler validation pipeline is built.

        Returns:
            tf.data.Dataset: The final, fully-configured dataset, ready to be
                             distributed to the TPUs.
        """
        #self.patch_sampler.set_hard_patchs(hard_patchs_list=hard_patchs_list)

        # self.patch_sampler.set_hard_patchs(hard_patchs_list)



        train_dataset = tf.data.Dataset.from_generator(
            self.genrator.train_data_genrator , 
            output_signature=(
                tf.TensorSpec(shape=self.config['data']['image_shape'], dtype=tf.float32),
                tf.TensorSpec(shape=self.config['data']['label_shape'], dtype=tf.int32),
                tf.RaggedTensorSpec(shape= [None , 3] , dtype = tf.int32)
            )                                         
        )
        options = tf.data.Options()
        options.experimental_deterministic = False
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        train_dataset = train_dataset.with_options(options)
        train_dataset = train_dataset.map(self.genrator.convert_to_tensor , num_parallel_calls=tf.data.AUTOTUNE)
        train_dataset = train_dataset.map(self.genrator.resize_volume , num_parallel_calls=tf.data.AUTOTUNE)
        train_dataset = train_dataset.map(self.genrator.cast , num_parallel_calls=tf.data.AUTOTUNE)
        if int(self.config['task']) == 1 : 
            train_dataset = train_dataset.map(
                lambda image, label, coord: (self._remap_labels(image, label)[0], self._remap_labels(image, label)[1], coord),
                num_parallel_calls=tf.data.AUTOTUNE
            )
        train_dataset = train_dataset.cache()
        train_dataset = train_dataset.map(self._geometric_augmentations  , num_parallel_calls=tf.data.AUTOTUNE)
        train_dataset = train_dataset.map(self._pad_volumes , num_parallel_calls=tf.data.AUTOTUNE)
        train_dataset = train_dataset.map(lambda image , label , hard_coords_for_volume_ragged  : self.patch_sampler.stage3_honring(image, label , hard_coords_for_volume_ragged) , num_parallel_calls=tf.data.AUTOTUNE)
        train_dataset = train_dataset.unbatch()
        train_dataset = train_dataset.map(self._intensity_augmentations , num_parallel_calls=tf.data.AUTOTUNE)
        train_dataset= train_dataset.shuffle(buffer_size=100).batch(self.final_batch_size , drop_remainder=True)
        train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

        val_dataset = tf.data.Dataset.from_generator(
            self.genrator.val_data_genrator ,
            output_signature=(
                tf.TensorSpec(shape=self.config['data']['image_shape'], dtype=tf.float32),
                tf.TensorSpec(shape=self.config['data']['label_shape'], dtype=tf.int32),
                tf.RaggedTensorSpec(shape= [None , 3] , dtype = tf.int32)
            )                                       
        ) 
        options = tf.data.Options()
        options.experimental_deterministic = False
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        val_dataset = val_dataset.with_options(options)
        val_dataset = val_dataset.map(self.genrator.convert_to_tensor, num_parallel_calls=tf.data.AUTOTUNE)
        val_dataset = val_dataset.map(self.genrator.resize_volume, num_parallel_calls=tf.data.AUTOTUNE)
        val_dataset = val_dataset.map(self.genrator.cast, num_parallel_calls=tf.data.AUTOTUNE)
        if int(self.config['task']) == 1 : 
            val_dataset = val_dataset.map(
                lambda image, label, coord: (self._remap_labels(image, label)[0], self._remap_labels(image, label)[1], coord),
                num_parallel_calls=tf.data.AUTOTUNE
            )
        val_dataset = val_dataset.cache()
        val_dataset = val_dataset.map(self._pad_volumes , num_parallel_calls=tf.data.AUTOTUNE)
        val_dataset = val_dataset.map(lambda image , label ,hard_coords_for_volume_ragged : self.patch_sampler.stage3_honring(image, label , hard_coords_for_volume_ragged ) , num_parallel_calls=tf.data.AUTOTUNE)
        val_dataset = val_dataset.unbatch()
        val_dataset = val_dataset.map(self._val_normalization , num_parallel_calls=tf.data.AUTOTUNE)
        val_dataset= val_dataset.batch(self.final_batch_size , drop_remainder=True)
        val_dataset = val_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

        return train_dataset , val_dataset

        