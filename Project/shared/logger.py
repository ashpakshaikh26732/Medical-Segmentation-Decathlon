
import sys
import os 
import tensorflow as tf
import nibabel as nib
import scipy
import numpy as np


from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy("mixed_float16")


# Add repo + Project folder to path
repo_path = "/content/drive/MyDrive/Medical-Segmentation-Decathlon"
sys.path.append(repo_path)
sys.path.append(os.path.join(repo_path, "Project"))



class Loading_and_saving_data:
    def __init__(self, image_address,label_address,record_path,loading_at_time =10):
        self.loading_at_time = loading_at_time
        self.image_address = image_address
        self.label_address = label_address
        self.record_path = record_path

    def load_and_read(self,batch,address) :
        train_batch_image = []
        for image in batch :
            img=os.path.join(address , image )
            img = nib.load(img)
            img = img.get_fdata()
            train_batch_image.append(img)
        return train_batch_image

    def convert_to_tensor(self , batch):
        train_batch = []
        for image in batch :
            img=tf.convert_to_tensor(image)
            train_batch.append(img)
        return train_batch

    def normalization(self, tensor_data): 
        mean = tf.math.reduce_mean(tensor_data ,axis=[0, 1, 2, 3], keepdims=True)
        std  = tf.math.reduce_std(tensor_data , axis=[0, 1, 2, 3], keepdims=True)
        tensor_data  = (tensor_data-mean)/(std + 1e-6) 
        return tensor_data

    def resize_volume_np(self,volume, interp_order,target_shape=(240,240,155)): 
        H,W,D,C=volume.shape
        target_h , target_w , target_d = target_shape
        zoom_factor = (target_h/H , target_w/W , target_d/D) 
        resized = np.zeros((target_h,target_w,target_d , C ))
        for c in range(C): 
            resized[...,c] = scipy.ndimage.zoom(volume[...,c],zoom_factor,order=interp_order)
        return resized

    def tf_resize_volume(self, volume , interp_order,target_shape=(240,240,155)):
        return tf.numpy_function(func = self.resize_volume_np,inp=[volume , interp_order,target_shape],Tout = tf.float32)

    def resize_batch_tensor(self,batch_tensor , interp_order,target_shape = (240,240,155)): 
        return tf.stack([self.tf_resize_volume(x , interp_order,target_shape)for x  in tf.unstack(batch_tensor)], axis = 0)

    def pad(self,data): 
        padding = tf.constant([
            [0,0],
            [0,0],
            [0,0],
            [0,1],
            [0,0]
        ])
        padded_data = tf.pad(data , padding = padding , mode='CONSTANT', constant_values=0) 
        return padded_data

    def make_patches(self, data ): 
        patches = []
        for x in range(0 , 240 , 80): 
            for y in range(0,240,80) : 
                for  z in range(0,156,52): 
                    patch = data[:, x:x+80, y:y+80, z:z+52, :]
                    patches.append(patch)

        patched_data = tf.stack (patches , axis = 1)
        return patched_data
    def cast(self,x,dtype = tf.float32): 
        return tf.cast(x, dtype=dtype)

    def _byte_feture(self , value): 
        return tf.train.Feature(bytes_list =tf.train.BytesList(value=[value])) 
    
    def _int64_feature(self, value): 
        return tf.train.Feature(int64_list  = tf.train.Int64List(value = [value]))

    def _string_feature(self, value ): 
        return tf.train.Feature(bytes_list = tf.train.BytesList(value = [value.encode()]))

    def save_patchwise_tfrecord(self, patched_images , patched_labels , volume_id , record_path): 

        for patch_id in range(patched_images.shape[0]): 
            full_record_path = os.path.join(record_path , f"volume{volume_id}_patch{patch_id}.tfrecord")
            with tf.io.TFRecordWriter(path=full_record_path) as writer : 
                image_raw = tf.io.serialize_tensor(patched_images[patch_id])
                label_raw = tf.io.serialize_tensor(patched_labels[patch_id])

                feature = { 
                    'image_patch' : self._byte_feture(image_raw.numpy()) , 
                    'label_patch': self._byte_feture(label_raw.numpy()) , 
                    'volume_id' : self._string_feature(volume_id) , 
                    'patch_id'  : self._int64_feature(patch_id)
                }
                example = tf.train.Example(Features = tf.train.Features(feature=feature))
                serilized_exmple = example.SerializeToString()

                writer.write(serilized_exmple)



    def load_for_preprocessing(self):
        images = sorted(os.listdir(self.image_address))
        labels = sorted(os.listdir(self.label_address))
        for i in range (0, len(images),self.loading_at_time):
            image_batch=images[i:i+self.loading_at_time]
            loaded_image_data=self.load_and_read(image_batch,self.image_address)
            image_tensor_data = self.convert_to_tensor(loaded_image_data)
            image_tensor_data = self.normalization(image_tensor_data)
            resized_image_tensor = self.resize_batch_tensor(image_tensor_data,interp_order=1)
            padded_image_tensor = self.pad(resized_image_tensor) 
            patched_image_tensors = self.make_patches(padded_image_tensor)
            casted_image_tensors = self.cast(patched_image_tensors) 

            label_batch = labels[i:i+self.loading_at_time]
            loaded_label_data = self.load_and_read(label_batch,self.label_address)
            label_tensor_data = self.convert_to_tensor(loaded_label_data) 
            resized_label_tensor = self.resize_batch_tensor(label_tensor_data , interp_order=0)
            padded_label_tensor = self.pad(resized_label_tensor)
            patched_label_tensors = self.make_patches(padded_label_tensor)
            casted_label_tensors = self.cast(patched_label_tensors) 

            for volume_id  in range (casted_image_tensors.shape[0]) : 
                self.save_patchwise_tfrecord(       
                    patched_images=casted_image_tensors[volume_id],
                    patched_labels=casted_label_tensors[volume_id],
                    volume_id=volume_id , 
                    record_path=self.record_path)


            
