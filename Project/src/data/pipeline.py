# import tensorflow as tf
# import os
# import nibabel as nib
# import scipy
# import numpy as np
# from scipy.ndimage import map_coordinates, gaussian_filter
# import sys

# from tensorflow.keras import mixed_precision
# mixed_precision.set_global_policy("mixed_float16")

# repo_path = "/content/drive/MyDrive/Medical-Segmentation-Decathlon"
# sys.path.append(repo_path)


# class DataPipeline:
#     def __init__(self, config , image_address , label_address):
#         self.val_count = config['data']['val_count']
#         self.image_address = image_address
#         self.label_address = label_address
#         self.num_classes = config['data']['num_classes']
#         self.final_batch_size = config['data']['batch'] * config['data']['num_replicas']
#         self.patch_shape = config['data']['patch_shape']
#         self.config = config

#     def load_and_read(self,data ) :

#         data = nib.load(data.numpy().decode('utf-8'))
#         data = data.get_fdata().astype(np.float32)

#         if len(data.shape) == 3:
#             data = np.expand_dims(data, axis=-1)

#         return data

#     def convert_to_tensor(self , image , label):

#         img=tf.convert_to_tensor(image)
#         label = tf.convert_to_tensor(label)
#         return img , label

#     def crop_or_pad_to_shape(self,volume, target_shape):
#         current_shape = volume.shape

#         for i in range(3):
#             if current_shape[i] > target_shape[i]:

#                 diff = current_shape[i] - target_shape[i]
#                 start = diff // 2
#                 end = start + target_shape[i]

#                 if i == 0: volume = volume[start:end, :, :, :]
#                 elif i == 1: volume = volume[:, start:end, :, :]
#                 else: volume = volume[:, :, start:end, :]
#             elif current_shape[i] < target_shape[i]:

#                 diff = target_shape[i] - current_shape[i]
#                 pad_before = diff // 2
#                 pad_after = diff - pad_before

#                 if i == 0: padding = ((pad_before, pad_after), (0, 0), (0, 0), (0, 0))
#                 elif i == 1: padding = ((0, 0), (pad_before, pad_after), (0, 0), (0, 0))
#                 else: padding = ((0, 0), (0, 0), (pad_before, pad_after), (0, 0))
#                 volume = np.pad(volume, padding, mode='constant', constant_values=0)
#         return volume

#     def elastic_transform_3d(self,  image, mask, alpha, sigma):

#         image = image.astype(np.float32)
#         mask = mask.astype(np.float32)
#         shape = image.shape
#         dx = gaussian_filter((np.random.rand(*shape[:3]) * 2 - 1), sigma) * alpha
#         dy = gaussian_filter((np.random.rand(*shape[:3]) * 2 - 1), sigma) * alpha
#         dz = gaussian_filter((np.random.rand(*shape[:3]) * 2 - 1), sigma) * alpha
#         h, w, d = shape[:3]
#         indices = np.meshgrid(np.arange(h), np.arange(w), np.arange(d), indexing='ij')
#         new_indices = [indices[0] + dx, indices[1] + dy, indices[2] + dz]
#         deformed_image = np.zeros_like(image)
#         deformed_mask = np.zeros_like(mask)
#         for c in range(shape[-1]):
#             deformed_image[..., c] = map_coordinates(image[..., c], new_indices, order=1, mode='reflect').reshape(shape[:3])
#         deformed_mask[..., 0] = map_coordinates(mask[..., 0], new_indices, order=0, mode='reflect').reshape(shape[:3])
#         return deformed_image, deformed_mask


#     def _geometric_augmentations_py(self,image, label):
#         """A pure Python/NumPy function for all geometric augmentations."""

#         original_shape = image.shape

#         if np.random.rand() > 0.5:
#             angle = np.random.uniform(-15, 15)

#             image = scipy.ndimage.rotate(image, angle=angle, axes=(1, 0), reshape=False, order=1)
#             label = scipy.ndimage.rotate(label, angle=angle, axes=(1, 0), reshape=False, order=0) #
#         if np.random.rand() > 0.5:
#             zoom_factor = np.random.uniform(0.9, 1.1)

#             image = scipy.ndimage.zoom(image, zoom=[zoom_factor, zoom_factor, 1, 1], order=1)
#             label = scipy.ndimage.zoom(label, zoom=[zoom_factor, zoom_factor, 1, 1], order=0)

#             image = self.crop_or_pad_to_shape(image, original_shape)
#             label = self.crop_or_pad_to_shape(label, original_shape)

#         if np.random.rand() > 0.5:
#             alpha = 15.0
#             sigma = 3.0
#             image, label = self.elastic_transform_3d(image, label, alpha, sigma)

#         return image, label


#     def tf_geometric_augmentations(self , image, label):
#         """TensorFlow wrapper for the geometric augmentations."""

#         aug_image, aug_label = tf.numpy_function(
#             func=self._geometric_augmentations_py,
#             inp=[image, label],
#             Tout=[tf.float32, tf.float32]
#         )


#         aug_image.set_shape(image.shape)
#         aug_label.set_shape(label.shape)

#         return aug_image, aug_label

#     def intensity_augmentations(self , image,label ):
#         """this is intensity augmentation """
#         min_value = tf.reduce_min(image)
#         max_value  =tf.reduce_max(image)
#         if self.config['data']['modality']=='MRI':
#             min_max_normalized_image = (image - min_value)/(max_value-min_value + 1e-8)

#         elif (self.config['data']['modality']) =='CT':
#             min_max_normalized_image = tf.clip_by_value(image , clip_value_min=-150 ,clip_value_max=250)
#             min_value = tf.reduce_min(min_max_normalized_image)
#             max_value = tf.reduce_max(min_max_normalized_image)
#             min_max_normalized_image = (min_max_normalized_image-min_value)/(max_value-min_value+1e-8)

#         image = tf.image.random_brightness(min_max_normalized_image , max_delta=0.1)
#         image = tf.image.random_contrast(image, lower=0.7 , upper=1.1)
#         Gaussian_noise = tf.random.normal((tf.shape(image)),stddev=0.1)
#         image = image + Gaussian_noise
#         gama_value = tf.random.uniform(() , minval=0.7 , maxval=1.5)
#         image = tf.clip_by_value(image, 0.0, 1.0)
#         image = tf.math.pow(image , gama_value)
#         image = tf.clip_by_value(image ,clip_value_min=0.0 , clip_value_max=1.0 )

#         mean = tf.math.reduce_mean(image ,axis=[1, 2, 3], keepdims=True)
#         std  = tf.math.reduce_std(image , axis=[1, 2, 3], keepdims=True)
#         image  = (image-mean)/(std + 1e-6)

#         return image , label


#     def resize_volume_np(self,volume, interp_order):
#         H,W,D,C=volume.shape
#         target_h , target_w , target_d , target_c = self.config['data']['image_shape']
#         zoom_factor = (target_h/H , target_w/W , target_d/D)
#         resized = np.zeros((target_h,target_w,target_d , C ),dtype =np.float32)
#         for c in range(C):
#             resized[...,c] = scipy.ndimage.zoom(volume[...,c],zoom_factor,order=interp_order)
#         return resized

#     def tf_resize_volume(self, volume , interp_order):
#         return tf.numpy_function(func = self.resize_volume_np,inp=[volume , interp_order],Tout = tf.float32)

#     def resized_image_and_label(self, image, label):
#         resized_image = self.tf_resize_volume(image, interp_order=1)
#         resized_label = self.tf_resize_volume(label, interp_order=0)


#         resized_image.set_shape(list(self.config['data']['image_shape']))
#         resized_label.set_shape(list(self.config['data']['label_shape']))

#         return resized_image, resized_label


#     def load(self , image , label):
#         image=tf.py_function(func=self.load_and_read , inp = [image],Tout=tf.float32)
#         label = tf.py_function(func = self.load_and_read , inp =[label] ,Tout=tf.float32)
#         return image , label

#     def padding(self, image, label):
#         paddings = [[0, 0] for _ in range(len(image.shape))]

#         for i in range(3):
#             image_dim = image.shape[i]
#             patch_dim = self.patch_shape[i]

#             if image_dim % patch_dim != 0:
#                 pad_amt = patch_dim - (image_dim % patch_dim)
#                 paddings[i][1] = pad_amt

#         padding = tf.constant (paddings)

#         padded_image = tf.pad(image , paddings=padding , mode = 'CONSTANT',constant_values=0)
#         padded_label = tf.pad(label , paddings=padding , mode='CONSTANT', constant_values=0)
#         return padded_image , padded_label

#     def make_patches(self,image , label):

#         image , label = self.padding(image , label)
#         image_patches = []
#         label_patches = []
#         h,w,d,c=image.shape
#         patch_h , patch_w, patch_d , patch_c = self.patch_shape
#         for x in range(0,h,patch_h):
#             for y in range(0,w,patch_w):
#                 for z in range(0, d, patch_d):
#                     patch = image[x:x+patch_h,y:y+patch_w , z:z+patch_d ,:]
#                     image_patches.append(patch)
#                     patch = label[x:x+patch_h,y:y+patch_w , z:z+patch_d ,:]
#                     label_patches.append(patch)
#         patched_image = tf.stack (image_patches , axis = 0)
#         patched_label = tf.stack(label_patches, axis=0)
#         return patched_image , patched_label

#     def val_normalization(self, image , label) : 
#         min_value = tf.reduce_min(image)
#         max_value  =tf.reduce_max(image)
#         if self.config['data']['modality']=='MRI':
#             min_max_normalized_image = (image - min_value)/(max_value-min_value + 1e-8)

#         elif (self.config['data']['modality']) =='CT':
#             min_max_normalized_image = tf.clip_by_value(image , clip_value_min=-150 ,clip_value_max=250)
#             min_value = tf.reduce_min(min_max_normalized_image)
#             max_value = tf.reduce_max(min_max_normalized_image)
#             min_max_normalized_image = (min_max_normalized_image-min_value)/(max_value-min_value+1e-8)


#         mean = tf.math.reduce_mean(min_max_normalized_image ,axis=[1, 2, 3], keepdims=True)
#         std  = tf.math.reduce_std(min_max_normalized_image , axis=[1, 2, 3], keepdims=True)
#         image  = (min_max_normalized_image-mean)/(std + 1e-6)

#         return image , label 


#     def cast(self,image , label,dtype = tf.float32):
#         image  =  tf.cast(image, dtype=dtype)
#         label = tf.cast(label , dtype = dtype)
#         return image , label
    
#     def remap_labels_for_braTS(self, x , y, target_num_classes=4):
#         """
#         Convert BraTS label 4 -> 3 so labels are sequential 0..3.
#         Input y can be shape [D,H,W] or [D,H,W,1] (or batched later).
#         Returns y with dtype tf.int32 and channel dim [-1] present.
#         """
#         original_dtype = y.dtype 
#         y = tf.cast(y, tf.int32)
#         # map official BraTS ET label 4 -> 3
#         y = tf.where(tf.equal(y, 4), tf.constant(3, dtype=tf.int32), y)
#         # clip any stray values to valid range 0..target_num_classes-1
#         y = tf.clip_by_value(y, 0, tf.cast(target_num_classes - 1, tf.int32))
#         # ensure channel dimension
#         if tf.rank(y) == 3:
#             y = tf.expand_dims(y, axis=-1)
#         y = tf.cast(y , dtype = original_dtype) 
#         return x , y


#     def getFullImageAddress(self,image):
#         return os.path.join(self.image_address,image)

#     def getFullLabelAddress(self,label):
#         return os.path.join(self.label_address,label)

#     def load_for_preprocessing(self):
#         global_id = 0
#         images = sorted([
#             f for f in os.listdir(self.image_address)
#             if f.endswith(".nii.gz") and not f.startswith("._")
#         ])

#         labels = sorted([
#             f for f in os.listdir(self.label_address)
#             if f.endswith(".nii.gz") and not f.startswith("._")
#         ])
#         val_images = images[:self.val_count]
#         val_labels = labels[:self.val_count]

#         train_images = images[self.val_count:]
#         train_labels = labels[self.val_count:]

#         train_images = list(map(self.getFullImageAddress , train_images))
#         train_labels = list(map(self.getFullLabelAddress , train_labels))
        
#         val_images = list(map(self.getFullImageAddress, val_images))
#         val_labels = list(map(self.getFullLabelAddress, val_labels))


#         train_dataset = tf.data.Dataset.from_tensor_slices((train_images , train_labels))
#         options = tf.data.Options()
#         options.experimental_deterministic = False
#         options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
#         train_dataset = train_dataset.with_options(options)
#         train_dataset =train_dataset.map(self.load , num_parallel_calls=tf.data.AUTOTUNE)
#         train_dataset = train_dataset.map(self.convert_to_tensor , num_parallel_calls=tf.data.AUTOTUNE)
#         train_dataset = train_dataset.map(self.resized_image_and_label,num_parallel_calls=tf.data.AUTOTUNE)
#         #if int(self.config['task']) ==1 : 
#         #    train_dataset = train_dataset.map(self.remap_labels_for_braTS , num_parallel_calls=tf.data.AUTOTUNE )

#         train_dataset = train_dataset.cache()
#         train_dataset = train_dataset.map(self.tf_geometric_augmentations , num_parallel_calls= tf.data.AUTOTUNE)
#         train_dataset = train_dataset.map(self.make_patches , num_parallel_calls=tf.data.AUTOTUNE)
#         train_dataset = train_dataset.unbatch()
#         train_dataset = train_dataset.map(self.cast, num_parallel_calls=tf.data.AUTOTUNE)
#         train_dataset = train_dataset.map(self.intensity_augmentations , num_parallel_calls = tf.data.AUTOTUNE)
#         train_dataset= train_dataset.shuffle(buffer_size=100).batch(self.final_batch_size , drop_remainder=True)
#         train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)



#         val_dataset = tf.data.Dataset.from_tensor_slices((val_images , val_labels))
#         options = tf.data.Options()
#         options.experimental_deterministic = False
#         options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

#         val_dataset = val_dataset.with_options(options)
#         val_dataset =val_dataset.map(self.load , num_parallel_calls=tf.data.AUTOTUNE)
#         val_dataset = val_dataset.map(self.convert_to_tensor , num_parallel_calls=tf.data.AUTOTUNE)
#         val_dataset = val_dataset.map(self.resized_image_and_label,num_parallel_calls=tf.data.AUTOTUNE)
#         #if int(self.config['task']) ==1 : 
#         #    val_dataset = val_dataset.map(self.remap_labels_for_braTS , num_parallel_calls=tf.data.AUTOTUNE )
        
#         val_dataset = val_dataset.cache()
#         val_dataset = val_dataset.map(self.make_patches , num_parallel_calls=tf.data.AUTOTUNE)
#         val_dataset = val_dataset.unbatch()
#         val_dataset = val_dataset.map(self.cast, num_parallel_calls=tf.data.AUTOTUNE)
#         val_dataset = val_dataset.map(self.val_normalization , num_parallel_calls=tf.data.AUTOTUNE)
#         val_dataset= val_dataset.batch(self.final_batch_size , drop_remainder=True)
#         val_dataset = val_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

#         return train_dataset , val_dataset




# new data pipeline 

import sys
import tensorflow as tf
import nibabel as nib
import os
import numpy as np
import tensorflow_graphics.geometry.transformation as tfg_transformation
import keras_cv


repo_path = "/content/drive/MyDrive/Medical-Segmentation-Decathlon"
sys.path.append(repo_path)

class Genrator :
    """
    A Python generator class for loading and performing initial preparation of 3D
    NIfTI medical imaging data on-the-fly.

    This class is designed to be the first stage of a high-performance data pipeline,
    handling the "out-of-graph" file I/O operations before passing data to TensorFlow.
    It reads .nii.gz files, splits them into training and validation sets, and yields
    them as NumPy arrays.
    """
    def __init__(self, config , image_address , label_address):
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

    def getFullImageAddress(self, image) :
        """Constructs the full path for an image filename."""
        return os.path.join(self.image_address,image)

    def getFullLabelAddress(self, label):
        """Constructs the full path for a label filename."""
        return os.path.join(self.label_address, label )

    def convert_to_tensor(self, image , label):
        """Converts NumPy arrays to TensorFlow tensors."""
        image = tf.convert_to_tensor(image)
        label = tf.convert_to_tensor(label)
        return image , label

    def resize_volume(self, image, label):
        """
        Resizes a 3D volume to a target shape specified in the config using
        the modern tf.image.resize function for robust interpolation.
        """
        # Get target shape from config, e.g., [D, H, W, C]
        target_image_shape = self.config['data']['image_shape']
        target_label_shape = self.config['data']['label_shape']

        # --- Resize Image (using 'bilinear' interpolation) ---
        # 1. Resize Height and Width: input shape (D, H, W, C)
        resized_hw = tf.image.resize(
            image,
            (target_image_shape[1], target_image_shape[2]),
            method='bilinear'
        )
        # 2. Transpose for depth resizing: new shape (H, W, D, C)
        transposed = tf.transpose(resized_hw, perm=[1, 2, 0, 3])
        # 3. Resize Depth: input shape (H, W, D, C), resizing the (W, D) plane
        resized_d = tf.image.resize(
            transposed,
            (target_image_shape[2], target_image_shape[0]), # Resizing W and D dims
            method='bilinear'
        )
        # 4. Transpose back to original format: final shape (D, H, W, C)
        final_image = tf.transpose(resized_d, perm=[2, 0, 1, 3])
        final_image.set_shape(target_image_shape)


        # --- Resize Label (using 'nearest' to preserve integer class IDs) ---
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

        return final_image, final_label

    def cast(self, image , label , dtype = tf.float32):
        """Casts image and label tensors to a specified data type."""
        image = tf.cast(image , dtype=dtype)
        label = tf.cast(label , dtype = dtype)
        return image , label

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

            yield image_volume , label_volume

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

            yield val_images_volume , val_labels_volume


class PatchSampler :
    def __init__(self , config):
        #self.patch_shape = config['data']['patch_shape']
        self.stage_patches_per_volume = config['data']['stage_patches_per_volume']
        self.stage1_fg_ratio = config['data']['stage1_fg_ratio']
        self.stage2_fg_ratio = config['data']['stage2_fg_ratio']

        self.stage2_intra_class_weights = list(config['data']['stage2_intra_class_ratio'])
        self.stage3_fg_ratio = config['data']['stage3_fg_ratio']

        self.stage3_hard_sample_ratio = config['data']['stage3_hard_sample_ratio']
        self.class_names = list(config['data']['class_names'])

        self.image_patch = list(config['data']['image_patch_shape'])

        self.label_patch = list(config['data']['label_patch_shape'])
        self.hard_patchs = None 
    
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




    # def _extract_patches(self, volume, centers, patch_shape):
    #     """
    #     Extracts 3D patches from a volume at specified center coordinates.

    #     This method is a fully graph-native, high-performance implementation using the
    #     SOTA "slice and gather" technique with tf.gather_nd.

    #     Args:
    #         volume (tf.Tensor): The 4D volume to extract from (D, H, W, C).
    #         centers (tf.Tensor): A tensor of center coordinates of shape [num_patches, 3].
    #         patch_shape (list or tuple): The spatial shape of the patches, e.g., (80, 80, 52).

    #     Returns:
    #         tf.Tensor: A tensor of extracted patches with shape 
    #                 [num_patches, patch_d, patch_h, patch_w, channels].
    #     """

    #     patch_shape = tf.convert_to_tensor(patch_shape[:3], dtype=tf.int32)
    #     centers = tf.cast(centers, dtype=tf.int32)

    #     half_patch = patch_shape // 2
    #     corner_coordinates = centers - half_patch


    #     zeros = tf.zeros_like(corner_coordinates)
    #     corner_coordinates = tf.maximum(corner_coordinates, zeros)

    #     patch_d, patch_h, patch_w = tf.unstack(patch_shape)
        
    #     z_indices = tf.range(patch_d)
    #     y_indices = tf.range(patch_h)
    #     x_indices = tf.range(patch_w)

    #     grid_z, grid_y, grid_x = tf.meshgrid(z_indices, y_indices, x_indices, indexing='ij')


    #     relative_patch_indices = tf.stack([grid_z, grid_y, grid_x], axis=-1)
    #     relative_patch_indices = tf.reshape(relative_patch_indices, (-1, 3))

    #     corner_coordinates = tf.expand_dims(corner_coordinates, axis=1)
    #     relative_patch_indices = tf.expand_dims(relative_patch_indices, axis=0)

    #     absolute_indices = corner_coordinates + relative_patch_indices

    #     patches_volume = tf.gather_nd(volume, absolute_indices)

    #     num_channels = tf.shape(volume)[-1]
    #     final_shape = tf.concat([
    #         [-1], 
    #         patch_shape, 
    #         [num_channels]
    #     ], axis=0)

    #     patches_volume = tf.reshape(patches_volume, shape=final_shape)
        
    #     return patches_volume

    def _sample_stage1_foundational(self, image, label):
        """
        Implements Stage 1: Foundational Learning with Foreground-Aware Sampling.

        This method uses the "Oversample with Replacement" strategy to guarantee
        a fixed foreground/random patch ratio in every batch, which is a core
        principle of the SOTA nnU-Net methodology.
        """
    
        num_fg_patches = tf.cast(self.stage_patches_per_volume * self.stage1_fg_ratio, dtype=tf.int32)
        num_random_patches = tf.cast(self.stage_patches_per_volume - num_fg_patches, dtype=tf.int32)


        all_fg_indices_with_channel = tf.where(label > 0)
        

        all_fg_indices = tf.cast(all_fg_indices_with_channel[:, :3], dtype=tf.int32)

        num_fg_found = tf.shape(all_fg_indices)[0]


        def sample_with_plenty():

            return tf.random.shuffle(all_fg_indices)[:num_fg_patches]

        def sample_with_few():

            random_indices = tf.random.uniform(shape=[num_fg_patches], maxval=tf.cast(num_fg_found, tf.int64), dtype=tf.int64)
            return tf.gather(all_fg_indices, random_indices)

        def sample_with_none():

            return tf.zeros(shape=(0, 3), dtype=tf.int32)


        foreground_centers = tf.cond(
            num_fg_found >= num_fg_patches,
            true_fn=sample_with_plenty,
            false_fn=lambda: tf.cond(
                num_fg_found > 0,
                true_fn=sample_with_few,
                false_fn=sample_with_none
            )
        )


        num_fg_sampled = tf.shape(foreground_centers)[0]
        num_total_random_patches = num_random_patches + (num_fg_patches - num_fg_sampled)


        volume_shape = tf.shape(image)
        max_d, max_h, max_w = volume_shape[0] - self.image_patch[0], volume_shape[1] - self.image_patch[1], volume_shape[2]-self.image_patch[2]

        random_d_coords = tf.random.uniform(shape=[num_total_random_patches], minval=0, maxval=max_d, dtype=tf.int32)
        random_h_coords = tf.random.uniform(shape=[num_total_random_patches], minval=0, maxval=max_h, dtype=tf.int32)
        random_w_coords = tf.random.uniform(shape=[num_total_random_patches], minval=0, maxval=max_w, dtype=tf.int32)

        random_centers = tf.stack([random_d_coords, random_h_coords, random_w_coords], axis=1)


        all_centers = tf.concat([foreground_centers, random_centers], axis=0)

        image_patchs = self._extract_patches( image , all_centers , self.image_patch )
        label_patchs = self._extract_patches( label , all_centers , self.label_patch )


        return image_patchs , label_patchs



    def _sample_stage2_refinement(self, image, label):
        """
        Implements Stage 2: Refinement with Probabilistic Rare-Class Oversampling.

        This method robustly handles all edge cases, including empty class pools,
        and uses TensorFlow-native loops (`tf.range`) to ensure it can be compiled
        into a high-performance graph.
        """
        num_fg_patches = tf.cast(self.stage_patches_per_volume * self.stage2_fg_ratio, dtype=tf.int32)
        num_random_patches = tf.cast(self.stage_patches_per_volume - num_fg_patches, dtype=tf.int32)


        class_coords = []
        non_empty_class_pools = []

        

        for i in range(1, len(self.class_names)):
            coords = tf.where(label == i)
            coords = tf.cast(coords[:, :3], dtype=tf.int32)
            class_coords.append(coords)
            if tf.shape(coords)[0] > 0:
                non_empty_class_pools.append(coords)


        all_fg_indices = tf.concat(class_coords, axis=0)


        def sample_with_foreground():

            
            class_logits = tf.math.log([self.stage2_intra_class_weights])

            chosen_class_indices = tf.random.categorical(logits=class_logits, num_samples=num_fg_patches)[0] 
            
            foreground_centers_ta = tf.TensorArray(dtype=tf.int32, size=num_fg_patches, dynamic_size=False)

            for i in tf.range(num_fg_patches):
                class_id_to_sample = chosen_class_indices[i]
                
                pool_of_coords = class_coords[tf.cast(class_id_to_sample, tf.int32)]
                num_coords_in_pool = tf.shape(pool_of_coords)[0]

                def pick_from_this_pool():

                    random_index = tf.random.uniform(shape=(), maxval=num_coords_in_pool, dtype=tf.int32)
                    return pool_of_coords[random_index]
                
                def pick_from_any_pool():

                    random_pool_index = tf.random.uniform(shape=(), maxval=len(non_empty_class_pools), dtype=tf.int32)
                    fallback_pool = non_empty_class_pools[random_pool_index]
                    random_coord_index = tf.random.uniform(shape=(), maxval=tf.shape(fallback_pool)[0], dtype=tf.int32)
                    return fallback_pool[random_coord_index]

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
        num_total_random_patches = num_random_patches + (num_fg_patches - num_fg_sampled)

        volume_shape = tf.shape(image)
        max_d, max_h, max_w = volume_shape[0] - self.image_patch[0], volume_shape[1] - self.image_patch[1], volume_shape[2]-self.image_patch[2]


        random_d_coords = tf.random.uniform(shape=[num_total_random_patches], minval=0, maxval=max_d, dtype=tf.int32)
        random_h_coords = tf.random.uniform(shape=[num_total_random_patches], minval=0, maxval=max_h, dtype=tf.int32)
        random_w_coords = tf.random.uniform(shape=[num_total_random_patches], minval=0, maxval=max_w, dtype=tf.int32)

        random_centers = tf.stack([random_d_coords, random_h_coords, random_w_coords], axis=1)
        
        foreground_centers.set_shape([None, 3])
        random_centers.set_shape([None, 3])
        
        all_centers = tf.concat([foreground_centers, random_centers], axis=0)
    
        image_patchs = self._extract_patches( image , all_centers , self.image_patch )
        label_patchs = self._extract_patches( label , all_centers , self.label_patch )
        
        return image_patchs , label_patchs 

    def set_hard_patchs(self, hard_patchs_list):
        """
        Public method to update the internal buffer of hard patch coordinates.
        This will be called periodically by the main training loop.
        """
        if hard_patchs_list == None : 
            return 

        self.hard_patchs = tf.convert_to_tensor(hard_patchs_list, dtype=tf.int32)
        print(f"INFO: Hard patch buffer updated with {tf.shape(self.hard_patchs)[0]} coordinates.")

    def _sample_stage3_hard_mining(self, image, label):
        """
        Implements Stage 3: Honing with Online Hard Patch Mining.

        This method samples from a pre-computed buffer of "hard" coordinates.
        If the buffer is empty, it robustly falls back to the Stage 1 strategy.
        """

        def fallback_to_stage1():

            print("WARNING: Hard patch buffer is empty. Falling back to Stage 1 sampling for this volume.")
            return self._sample_stage1_foundational(image, label)

        def sample_from_hard_buffer():

            num_hard_patches = tf.cast(self.stage_patches_per_volume * self.stage3_hard_sample_ratio, dtype=tf.int32)
            num_random_patches = tf.cast(self.stage_patches_per_volume - num_hard_patches, dtype=tf.int32)

            num_available_hard = tf.shape(self.hard_patchs)[0]

            random_indices = tf.random.uniform(shape=[num_hard_patches], maxval=tf.cast(num_available_hard, tf.int64), dtype=tf.int64)
            hard_patch_coords = tf.gather(self.hard_patchs, random_indices)

            volume_shape = tf.shape(image)
            max_d, max_h, max_w = volume_shape[0] - self.image_patch[0], volume_shape[1] - self.image_patch[1], volume_shape[2]-self.image_patch[2]


            random_d_coords = tf.random.uniform(shape=[num_random_patches], minval=0, maxval=max_d, dtype=tf.int32)
            random_h_coords = tf.random.uniform(shape=[num_random_patches], minval=0, maxval=max_h, dtype=tf.int32)
            random_w_coords = tf.random.uniform(shape=[num_random_patches], minval=0, maxval=max_w, dtype=tf.int32)

            random_centers = tf.stack([random_d_coords, random_h_coords, random_w_coords], axis=1)

            all_centers = tf.concat([hard_patch_coords, random_centers], axis=0)
            return all_centers

        all_centers = tf.cond(
            self.hard_patchs is None or tf.shape(self.hard_patchs)[0] == 0,
            true_fn=fallback_to_stage1,
            false_fn=sample_from_hard_buffer
        )

        image_patches = self._extract_patches(image, all_centers, self.image_patch_shape)
        label_patches = self._extract_patches(label, all_centers, self.label_patch_shape)

        return image_patches, label_patches

    def sample(self, image , label ,stage ): 
        """
        Public-facing router method to dispatch to the correct sampling strategy.

        This is the main entry point for the DataPipeline. It takes the current
        training stage as an argument and calls the appropriate private method
        to generate a batch of patches.

        Args:
            image (tf.Tensor): The full 4D image volume.
            label (tf.Tensor): The full 4D label volume.
            stage (str): A string identifier for the current training stage, e.g.,
                         'stage1_foundational', 'stage2_refinement'.

        Returns:
            Tuple[tf.Tensor, tf.Tensor]: A tuple containing the final tensors for
            the extracted image patches and label patches.
        """

        if stage == 'stage1_foundational':
            return self._sample_stage1_foundational(image, label)
        elif stage == 'stage2_refinement':
            return self._sample_stage2_refinement(image, label)
        elif stage == 'stage3_hard_mining':
            return self._sample_stage3_hard_mining(image, label)
        else:
            print(f"Warning: Unknown strategy '{stage}'. Defaulting to Stage 1.")
            return self._sample_stage1_foundational(image, label)



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

class DataPipeline : 
    def __init__(self, config , image_address , label_address):
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
        num_classes = config['data']['num_classes']
        self.config = config 
        self.genrator = Genrator(config , image_address=image_address , label_address=label_address)
        self.patch_sampler = PatchSampler(config)
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
        self.patch_shape = list (config['data']['image_patch_shape']) 
        self.final_batch_size = config['data']['batch'] * config['data']['num_replicas']
    
    def _geometric_augmentations(self, image, label):
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

        return image , label 

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
        min_value = tf.reduce_min(image)
        max_value  =tf.reduce_max(image)
        if self.config['data']['modality']=='MRI':
            min_max_normalized_image = (image - min_value)/(max_value-min_value + 1e-8)

        elif (self.config['data']['modality']) =='CT':
            min_max_normalized_image = tf.clip_by_value(image , clip_value_min=-150 ,clip_value_max=250)
            min_value = tf.reduce_min(min_max_normalized_image)
            max_value = tf.reduce_max(min_max_normalized_image)
            min_max_normalized_image = (min_max_normalized_image-min_value)/(max_value-min_value+1e-8)

        image = tf.image.random_brightness(min_max_normalized_image , max_delta=0.1)
        image = tf.image.random_contrast(image, lower=0.7 , upper=1.1)
        Gaussian_noise = tf.random.normal((tf.shape(image)),stddev=0.1 , dtype = image.dtype)
        image = image + Gaussian_noise
        gama_value = tf.random.uniform(() , minval=0.7 , maxval=1.5 ,  dtype=image.dtype)
        image = tf.clip_by_value(image, 0.0, 1.0)
        image = tf.math.pow(image , gama_value)
        image = tf.clip_by_value(image ,clip_value_min=0.0 , clip_value_max=1.0 )

        mean = tf.math.reduce_mean(image ,axis=[1, 2, 3], keepdims=True)
        std  = tf.math.reduce_std(image , axis=[1, 2, 3], keepdims=True)
        image  = (image-mean)/(std + 1e-6)

        return image , label

    def _pad_volumes(self, image_volume, label_volume):
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

        return padded_image, padded_label
    
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
        min_value = tf.reduce_min(image)
        max_value  =tf.reduce_max(image)
        if self.config['data']['modality']=='MRI':
            min_max_normalized_image = (image - min_value)/(max_value-min_value + 1e-8)

        elif (self.config['data']['modality']) =='CT':
            min_max_normalized_image = tf.clip_by_value(image , clip_value_min=-150 ,clip_value_max=250)
            min_value = tf.reduce_min(min_max_normalized_image)
            max_value = tf.reduce_max(min_max_normalized_image)
            min_max_normalized_image = (min_max_normalized_image-min_value)/(max_value-min_value+1e-8)


        mean = tf.math.reduce_mean(min_max_normalized_image ,axis=[1, 2, 3], keepdims=True)
        std  = tf.math.reduce_std(min_max_normalized_image , axis=[1, 2, 3], keepdims=True)
        image  = (min_max_normalized_image-mean)/(std + 1e-6)

        return image , label 


    def get_dataset(self, stage = 'stage1_foundational', hard_patchs_list = None,is_training = True ): 
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
        train_dataset = tf.data.Dataset.from_generator(
            self.genrator.train_data_genrator , 
            output_types=(tf.float32 , tf.int32) , 
            output_shapes = (tf.TensorShape(list(self.config['data']['image_shape'])),
                             tf.TensorShape(list(self.config['data']['label_shape']))
                             )                                           
                            )
        options = tf.data.Options()
        options.experimental_deterministic = False
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        train_dataset = train_dataset.with_options(options)
        train_dataset = train_dataset.map(self.genrator.convert_to_tensor , num_parallel_calls=tf.data.AUTOTUNE)
        train_dataset = train_dataset.map(self.genrator.resize_volume , num_parallel_calls=tf.data.AUTOTUNE)
        train_dataset = train_dataset.map(self.genrator.cast , num_parallel_calls=tf.data.AUTOTUNE)
        train_dataset = train_dataset.cache()
        train_dataset = train_dataset.map(self._geometric_augmentations  , num_parallel_calls=tf.data.AUTOTUNE)
        train_dataset = train_dataset.map(self._pad_volumes , num_parallel_calls=tf.data.AUTOTUNE)
        train_dataset = train_dataset.map(lambda image , label  : self.patch_sampler.sample(image, label ,stage ) , num_parallel_calls=tf.data.AUTOTUNE)
        train_dataset = train_dataset.unbatch()
        train_dataset = train_dataset.map(self._intensity_augmentations , num_parallel_calls=tf.data.AUTOTUNE)
        train_dataset= train_dataset.shuffle(buffer_size=100).batch(self.final_batch_size , drop_remainder=True)
        train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

        val_dataset = tf.data.Dataset.from_generator(
            self.genrator.val_data_genrator ,
            output_types=(tf.float32 , tf.int32) , 
            output_shapes = (tf.TensorShape(list(self.config['data']['image_shape'])),
                             tf.TensorShape(list(self.config['data']['label_shape']))
                             )                                                       
                            ) 
        options = tf.data.Options()
        options.experimental_deterministic = False
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        val_dataset = val_dataset.with_options(options)
        val_dataset = val_dataset.map(self.genrator.convert_to_tensor, num_parallel_calls=tf.data.AUTOTUNE)
        val_dataset = val_dataset.map(self.genrator.resize_volume, num_parallel_calls=tf.data.AUTOTUNE)
        val_dataset = val_dataset.map(self.genrator.cast, num_parallel_calls=tf.data.AUTOTUNE)
        val_dataset = val_dataset.cache()
        val_dataset = val_dataset.map(self._pad_volumes , num_parallel_calls=tf.data.AUTOTUNE)
        val_dataset = val_dataset.map(lambda image , label  : self.patch_sampler.sample(image, label ,stage ) , num_parallel_calls=tf.data.AUTOTUNE)
        val_dataset = val_dataset.unbatch()
        val_dataset = val_dataset.map(self._val_normalization , num_parallel_calls=tf.data.AUTOTUNE)
        val_dataset= val_dataset.batch(self.final_batch_size , drop_remainder=True)
        val_dataset = val_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

        return train_dataset , val_dataset

        