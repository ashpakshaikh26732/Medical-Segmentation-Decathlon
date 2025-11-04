import tensorflow as tf
import nibabel as nib
import os
import sys
import numpy as np 

repo_path = "/content/drive/MyDrive/Medical-Segmentation-Decathlon"
sys.path.append(repo_path)

class Mining_Data_Generator:
    """
    A Python generator for loading 3D NIfTI medical imaging data for the OHEM mining phase.

    This class handles the out-of-graph file I/O, reading .nii.gz files and
    yielding the full 3D volumes along with their file paths, which act as
    a unique volume identifier.
    """
    def __init__(self, config, image_address, label_address):
        """
        Initializes the generator.

        Args:
            config (dict): The main configuration dictionary.
            image_address (str): Path to the directory with image files.
            label_address (str): Path to the directory with label files.
        """
        self.config = config
        self.image_address = image_address
        self.label_address = label_address
        self.val_count = config['data']['val_count']

    def get_full_image_address(self, image):
        """Constructs the full path for an image filename."""
        return os.path.join(self.image_address, image)

    def get_full_label_address(self, label):
        """Constructs the full path for a label filename."""
        return os.path.join(self.label_address, label)

    def load_data(self, path: str) -> np.ndarray:
        """Loads a single NIfTI file into a 4D NumPy array."""
        data = nib.load(path).get_fdata().astype(np.float32)
        if len(data.shape) == 3:
            data = np.expand_dims(data, axis=-1)
        return data

    def get_data_paths(self):
        """Gets sorted lists of image and label filenames for the training split."""
        images = sorted([f for f in os.listdir(self.image_address) if f.endswith(".nii.gz") and not f.startswith("._")])
        labels = sorted([f for f in os.listdir(self.label_address) if f.endswith(".nii.gz") and not f.startswith("._")])
        return images[self.val_count:], labels[self.val_count:]

    def __call__(self):
        """
        A Python generator that yields data for the mining pipeline.

        For each volume in the training set, it loads the image and label data
        and yields them along with their original file paths.

        Yields:
            tuple: A tuple containing (image_volume, label_volume, image_path, label_path).
        """
        train_images_names, train_labels_names = self.get_data_paths()
        train_images_paths = [self.get_full_image_address(name) for name in train_images_names]
        train_labels_paths = [self.get_full_label_address(name) for name in train_labels_names]

        for img_path, label_path in zip(train_images_paths, train_labels_paths):
            image_volume = self.load_data(img_path)
            label_volume = self.load_data(label_path)
            yield image_volume, label_volume, img_path, label_path

class PatchSamplerOHEMMiner:
    """
    Handles the systematic, grid-based extraction of patches for the OHEM mining phase.
    """
    def __init__(self, config):
        """
        Initializes the patch sampler.

        Args:
            config (dict): The main configuration dictionary.
        """
        self.config = config
        self.patch_shape = config['data']['image_patch_shape'] # Corrected key

    def _generate_patch_coordinates(self, grid_dims):
        """
        Generates a tensor of 3D coordinates for each patch in a grid.

        Args:
            grid_dims (tf.Tensor): A 1D tensor of shape (3,) representing the
                                   grid dimensions [grid_d, grid_h, grid_w].

        Returns:
            tf.Tensor: A 2D tensor of shape [num_patches, 3] containing the
                       (d, h, w) coordinate for each patch.
        """
        gd, gh, gw = grid_dims[0], grid_dims[1], grid_dims[2]
        zz, yy, xx = tf.meshgrid(tf.range(gd), tf.range(gh), tf.range(gw), indexing='ij')
        coords = tf.stack([zz, yy, xx], axis=-1)
        return tf.reshape(coords, [-1, 3])

    def __call__(self, image, label, img_path, lbl_path):
        """
        Extracts all non-overlapping patches from a volume and associates metadata.

        This function performs the SOTA "grid-extract" operation and generates
        the complete metadata (volume identifier + patch coordinate) needed for mining.

        Args:
            image (tf.Tensor): The full 4D image volume.
            label (tf.Tensor): The full 4D label volume.
            img_path (tf.Tensor): A string tensor with the file path for the image.
            lbl_path (tf.Tensor): A string tensor with the file path for the label.

        Returns:
            A tuple of tensors: (image_patches, label_patches, repeated_img_paths, patch_coords).
        """

        ksizes = [1, self.patch_shape[0], self.patch_shape[1], self.patch_shape[2], 1]
        strides = [1, self.patch_shape[0], self.patch_shape[1], self.patch_shape[2], 1]


        image_batched = tf.expand_dims(image, axis=0)
        label_batched = tf.expand_dims(label, axis=0)


        image_patches_flat = tf.extract_volume_patches(image_batched, ksizes=ksizes, strides=strides, padding='SAME')
        label_patches_flat = tf.extract_volume_patches(label_batched, ksizes=ksizes, strides=strides, padding='SAME')


        grid_dims = tf.shape(image_patches_flat)[1:4]
        num_patches = grid_dims[0] * grid_dims[1] * grid_dims[2]
        num_image_channels = tf.shape(image)[-1]
        num_label_channels = tf.shape(label)[-1]

        image_patches = tf.reshape(image_patches_flat, [num_patches, self.patch_shape[0], self.patch_shape[1], self.patch_shape[2], num_image_channels])
        label_patches = tf.reshape(label_patches_flat, [num_patches, self.patch_shape[0], self.patch_shape[1], self.patch_shape[2], num_label_channels])


        patch_coords = self._generate_patch_coordinates(grid_dims)


        repeated_img_paths = tf.tile(tf.expand_dims(img_path, axis=0), [num_patches])


        return image_patches, label_patches, repeated_img_paths, patch_coords

class OHEMDataPipeline:
    """
    Orchestrates the creation of the tf.data.Dataset for the OHEM mining phase.
    """
    def __init__(self, config, image_address, label_address):
        """
        Initializes the pipeline orchestrator.
        """
        self.config = config

        self.generator = Mining_Data_Generator(config, image_address, label_address)
        self.patch_sampler = PatchSamplerOHEMMiner(config)
        self.final_batch_size = config['data']['batch'] * config['data']['num_replicas']
    
    def normalization(self, image , label) : 
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
    
    def get_mining_dataset(self):
        """
        Builds and returns the complete, non-augmented, non-shuffled dataset
        for OHEM, yielding patches and their full metadata.
        """
        mining_dataset = tf.data.Dataset.from_generator(
            self.generator,
            output_signature=(
                tf.TensorSpec(shape=self.config['data']['image_shape'], dtype=tf.float32),
                tf.TensorSpec(shape=self.config['data']['label_shape'], dtype=tf.int32),
                tf.TensorSpec(shape=(), dtype=tf.string),
                tf.TensorSpec(shape=(), dtype=tf.string)
            )
        )

        options = tf.data.Options()
        options.experimental_deterministic = False
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        mining_dataset = mining_dataset.with_options(options)


        mining_dataset = mining_dataset.flat_map(
            lambda image, label, img_path, lbl_path: tf.data.Dataset.from_tensor_slices(
                self.patch_sampler(image, label, img_path, lbl_path)
            )
        )
        tpu_dataset = mining_dataset.map(lambda image , label, img_path , label_path : (image, label)).map(self.normalization  ,num_parallel_calls=tf.data.AUTOTUNE ).map(self._remap_labels,num_parallel_calls=tf.data.AUTOTUNE).batch(self.final_batch_size,drop_remainder=True).prefetch(buffer_size=tf.data.AUTOTUNE)
        metadata_dataset = mining_dataset.map(lambda image , label , img_path , label_path : (img_path , label_path)).batch(self.final_batch_size, drop_remainder=True).prefetch(buffer_size=tf.data.AUTOTUNE)

        return tpu_dataset ,metadata_dataset 