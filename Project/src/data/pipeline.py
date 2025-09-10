import tensorflow as tf
import os
import nibabel as nib
import scipy
import numpy as np
from scipy.ndimage import map_coordinates, gaussian_filter
import sys

from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy("mixed_float16")

repo_path = "/content/drive/MyDrive/Medical-Segmentation-Decathlon"
sys.path.append(repo_path)


class DataPipeline:
    def __init__(self, config , image_address , label_address):
        self.val_count = config['data']['val_count']
        self.image_address = image_address
        self.label_address = label_address
        self.num_classes = config['data']['num_classes']
        self.final_batch_size = config['data']['batch'] * config['data']['num_replicas']
        self.patch_shape = config['data']['patch_shape']
        self.config = config

    def load_and_read(self,data ) :

        data = nib.load(data.numpy().decode('utf-8'))
        data = data.get_fdata().astype(np.float32)

        if len(data.shape) == 3:
            data = np.expand_dims(data, axis=-1)

        return data

    def convert_to_tensor(self , image , label):

        img=tf.convert_to_tensor(image)
        label = tf.convert_to_tensor(label)
        return img , label

    def crop_or_pad_to_shape(self,volume, target_shape):
        current_shape = volume.shape

        for i in range(3):
            if current_shape[i] > target_shape[i]:

                diff = current_shape[i] - target_shape[i]
                start = diff // 2
                end = start + target_shape[i]

                if i == 0: volume = volume[start:end, :, :, :]
                elif i == 1: volume = volume[:, start:end, :, :]
                else: volume = volume[:, :, start:end, :]
            elif current_shape[i] < target_shape[i]:

                diff = target_shape[i] - current_shape[i]
                pad_before = diff // 2
                pad_after = diff - pad_before

                if i == 0: padding = ((pad_before, pad_after), (0, 0), (0, 0), (0, 0))
                elif i == 1: padding = ((0, 0), (pad_before, pad_after), (0, 0), (0, 0))
                else: padding = ((0, 0), (0, 0), (pad_before, pad_after), (0, 0))
                volume = np.pad(volume, padding, mode='constant', constant_values=0)
        return volume

    def elastic_transform_3d(self,  image, mask, alpha, sigma):

        image = image.astype(np.float32)
        mask = mask.astype(np.float32)
        shape = image.shape
        dx = gaussian_filter((np.random.rand(*shape[:3]) * 2 - 1), sigma) * alpha
        dy = gaussian_filter((np.random.rand(*shape[:3]) * 2 - 1), sigma) * alpha
        dz = gaussian_filter((np.random.rand(*shape[:3]) * 2 - 1), sigma) * alpha
        h, w, d = shape[:3]
        indices = np.meshgrid(np.arange(h), np.arange(w), np.arange(d), indexing='ij')
        new_indices = [indices[0] + dx, indices[1] + dy, indices[2] + dz]
        deformed_image = np.zeros_like(image)
        deformed_mask = np.zeros_like(mask)
        for c in range(shape[-1]):
            deformed_image[..., c] = map_coordinates(image[..., c], new_indices, order=1, mode='reflect').reshape(shape[:3])
        deformed_mask[..., 0] = map_coordinates(mask[..., 0], new_indices, order=0, mode='reflect').reshape(shape[:3])
        return deformed_image, deformed_mask


    def _geometric_augmentations_py(self,image, label):
        """A pure Python/NumPy function for all geometric augmentations."""

        original_shape = image.shape

        if np.random.rand() > 0.5:
            angle = np.random.uniform(-15, 15)

            image = scipy.ndimage.rotate(image, angle=angle, axes=(1, 0), reshape=False, order=1)
            label = scipy.ndimage.rotate(label, angle=angle, axes=(1, 0), reshape=False, order=0) #
        if np.random.rand() > 0.5:
            zoom_factor = np.random.uniform(0.9, 1.1)

            image = scipy.ndimage.zoom(image, zoom=[zoom_factor, zoom_factor, 1, 1], order=1)
            label = scipy.ndimage.zoom(label, zoom=[zoom_factor, zoom_factor, 1, 1], order=0)

            image = self.crop_or_pad_to_shape(image, original_shape)
            label = self.crop_or_pad_to_shape(label, original_shape)

        if np.random.rand() > 0.5:
            alpha = 15.0
            sigma = 3.0
            image, label = self.elastic_transform_3d(image, label, alpha, sigma)

        return image, label


    def tf_geometric_augmentations(self , image, label):
        """TensorFlow wrapper for the geometric augmentations."""

        aug_image, aug_label = tf.numpy_function(
            func=self._geometric_augmentations_py,
            inp=[image, label],
            Tout=[tf.float32, tf.float32]
        )


        aug_image.set_shape(image.shape)
        aug_label.set_shape(label.shape)

        return aug_image, aug_label

    def intensity_augmentations(self , image,label ):
        """this is intensity augmentation """
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
        Gaussian_noise = tf.random.normal((tf.shape(image)),stddev=0.1)
        image = image + Gaussian_noise
        gama_value = tf.random.uniform(() , minval=0.7 , maxval=1.5)
        image = tf.clip_by_value(image, 0.0, 1.0)
        image = tf.math.pow(image , gama_value)
        image = tf.clip_by_value(image ,clip_value_min=0.0 , clip_value_max=1.0 )

        mean = tf.math.reduce_mean(image ,axis=[1, 2, 3], keepdims=True)
        std  = tf.math.reduce_std(image , axis=[1, 2, 3], keepdims=True)
        image  = (image-mean)/(std + 1e-6)

        return image , label


    def resize_volume_np(self,volume, interp_order):
        H,W,D,C=volume.shape
        target_h , target_w , target_d , target_c = self.config['data']['image_shape']
        zoom_factor = (target_h/H , target_w/W , target_d/D)
        resized = np.zeros((target_h,target_w,target_d , C ),dtype =np.float32)
        for c in range(C):
            resized[...,c] = scipy.ndimage.zoom(volume[...,c],zoom_factor,order=interp_order)
        return resized

    def tf_resize_volume(self, volume , interp_order):
        return tf.numpy_function(func = self.resize_volume_np,inp=[volume , interp_order],Tout = tf.float32)

    def resized_image_and_label(self, image, label):
        resized_image = self.tf_resize_volume(image, interp_order=1)
        resized_label = self.tf_resize_volume(label, interp_order=0)


        resized_image.set_shape(list(self.config['data']['image_shape']))
        resized_label.set_shape(list(self.config['data']['label_shape']))

        return resized_image, resized_label


    def load(self , image , label):
        image=tf.py_function(func=self.load_and_read , inp = [image],Tout=tf.float32)
        label = tf.py_function(func = self.load_and_read , inp =[label] ,Tout=tf.float32)
        return image , label

    def padding(self, image, label):
        paddings = [[0, 0] for _ in range(len(image.shape))]

        for i in range(3):
            image_dim = image.shape[i]
            patch_dim = self.patch_shape[i]

            if image_dim % patch_dim != 0:
                pad_amt = patch_dim - (image_dim % patch_dim)
                paddings[i][1] = pad_amt

        padding = tf.constant (paddings)

        padded_image = tf.pad(image , paddings=padding , mode = 'CONSTANT',constant_values=0)
        padded_label = tf.pad(label , paddings=padding , mode='CONSTANT', constant_values=0)
        return padded_image , padded_label

    def make_patches(self,image , label):

        image , label = self.padding(image , label)
        image_patches = []
        label_patches = []
        h,w,d,c=image.shape
        patch_h , patch_w, patch_d , patch_c = self.patch_shape
        for x in range(0,h,patch_h):
            for y in range(0,w,patch_w):
                for z in range(0, d, patch_d):
                    patch = image[x:x+patch_h,y:y+patch_w , z:z+patch_d ,:]
                    image_patches.append(patch)
                    patch = label[x:x+patch_h,y:y+patch_w , z:z+patch_d ,:]
                    label_patches.append(patch)
        patched_image = tf.stack (image_patches , axis = 0)
        patched_label = tf.stack(label_patches, axis=0)
        return patched_image , patched_label

    def val_normalization(self, image , label) : 
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


    def cast(self,image , label,dtype = tf.float32):
        image  =  tf.cast(image, dtype=dtype)
        label = tf.cast(label , dtype = dtype)
        return image , label
    
    def remap_labels_for_braTS(self, x , y, target_num_classes=4):
        """
        Convert BraTS label 4 -> 3 so labels are sequential 0..3.
        Input y can be shape [D,H,W] or [D,H,W,1] (or batched later).
        Returns y with dtype tf.int32 and channel dim [-1] present.
        """
        original_dtype = y.dtype 
        y = tf.cast(y, tf.int32)
        # map official BraTS ET label 4 -> 3
        y = tf.where(tf.equal(y, 4), tf.constant(3, dtype=tf.int32), y)
        # clip any stray values to valid range 0..target_num_classes-1
        y = tf.clip_by_value(y, 0, tf.cast(target_num_classes - 1, tf.int32))
        # ensure channel dimension
        if tf.rank(y) == 3:
            y = tf.expand_dims(y, axis=-1)
        y = tf.cast(y , dtype = original_dtype) 
        return x , y


    def getFullImageAddress(self,image):
        return os.path.join(self.image_address,image)

    def getFullLabelAddress(self,label):
        return os.path.join(self.label_address,label)

    def load_for_preprocessing(self):
        global_id = 0
        images = sorted([
            f for f in os.listdir(self.image_address)
            if f.endswith(".nii.gz") and not f.startswith("._")
        ])

        labels = sorted([
            f for f in os.listdir(self.label_address)
            if f.endswith(".nii.gz") and not f.startswith("._")
        ])
        val_images = images[:self.val_count]
        val_labels = labels[:self.val_count]

        train_images = images[self.val_count:]
        train_labels = labels[self.val_count:]

        train_images = list(map(self.getFullImageAddress , train_images))
        train_labels = list(map(self.getFullLabelAddress , train_labels))
        
        val_images = list(map(self.getFullImageAddress, val_images))
        val_labels = list(map(self.getFullLabelAddress, val_labels))


        train_dataset = tf.data.Dataset.from_tensor_slices((train_images , train_labels))
        options = tf.data.Options()
        options.experimental_deterministic = False
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        train_dataset = train_dataset.with_options(options)
        train_dataset =train_dataset.map(self.load , num_parallel_calls=tf.data.AUTOTUNE)
        train_dataset = train_dataset.map(self.convert_to_tensor , num_parallel_calls=tf.data.AUTOTUNE)
        train_dataset = train_dataset.map(self.resized_image_and_label,num_parallel_calls=tf.data.AUTOTUNE)
        #if int(self.config['task']) ==1 : 
        #    train_dataset = train_dataset.map(self.remap_labels_for_braTS , num_parallel_calls=tf.data.AUTOTUNE )

        train_dataset = train_dataset.cache()
        train_dataset = train_dataset.map(self.tf_geometric_augmentations , num_parallel_calls= tf.data.AUTOTUNE)
        train_dataset = train_dataset.map(self.make_patches , num_parallel_calls=tf.data.AUTOTUNE)
        train_dataset = train_dataset.unbatch()
        train_dataset = train_dataset.map(self.cast, num_parallel_calls=tf.data.AUTOTUNE)
        train_dataset = train_dataset.map(self.intensity_augmentations , num_parallel_calls = tf.data.AUTOTUNE)
        train_dataset= train_dataset.shuffle(buffer_size=100).batch(self.final_batch_size , drop_remainder=True)
        train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)



        val_dataset = tf.data.Dataset.from_tensor_slices((val_images , val_labels))
        options = tf.data.Options()
        options.experimental_deterministic = False
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

        val_dataset = val_dataset.with_options(options)
        val_dataset =val_dataset.map(self.load , num_parallel_calls=tf.data.AUTOTUNE)
        val_dataset = val_dataset.map(self.convert_to_tensor , num_parallel_calls=tf.data.AUTOTUNE)
        val_dataset = val_dataset.map(self.resized_image_and_label,num_parallel_calls=tf.data.AUTOTUNE)
        #if int(self.config['task']) ==1 : 
        #    val_dataset = val_dataset.map(self.remap_labels_for_braTS , num_parallel_calls=tf.data.AUTOTUNE )
        
        val_dataset = val_dataset.cache()
        val_dataset = val_dataset.map(self.make_patches , num_parallel_calls=tf.data.AUTOTUNE)
        val_dataset = val_dataset.unbatch()
        val_dataset = val_dataset.map(self.cast, num_parallel_calls=tf.data.AUTOTUNE)
        val_dataset = val_dataset.map(self.val_normalization , num_parallel_calls=tf.data.AUTOTUNE)
        val_dataset= val_dataset.batch(self.final_batch_size , drop_remainder=True)
        val_dataset = val_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

        return train_dataset , val_dataset

