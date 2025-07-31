import sys, os
import tensorflow as tf 

from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy("mixed_float16")

repo_path = "/content/drive/MyDrive/Medical-Segmentation-Decathlon"
sys.path.append(repo_path)

class diceLoss(tf.keras.losses.Loss):
    def __init__(self, smooth=1e-6,num_classes =3):
        super().__init__(reduction = tf.keras.losses.Reduction.NONE)
        self.smooth = smooth
        self.num_classes = num_classes
    def compute_per_sample(self, y_true, y_pred):
        dtype=y_pred.dtype

        y_true = tf.one_hot(y_true, depth=self.num_classes,dtype=dtype)
        intersection = tf.reduce_sum(y_true*y_pred , axis=[1,2,3])
        union = tf.reduce_sum(y_true, axis = [1,2,3]) + tf.reduce_sum(y_pred, axis = [1,2,3])
        dice_per_class = 2*(intersection+self.smooth)/(union +self.smooth)
        dice_loss= 1 -tf.reduce_mean(dice_per_class , axis = -1)
        return dice_loss
    def call (self,y_true , y_pred):
        """fallback for model.fit"""
        return tf.reduce_mean(self.compute_per_sample(y_true,y_pred))


class Sementic_segmentation_loss(tf.keras.losses.Loss):
    def __init__(self , alpha=1.0 , beta=1.0  , smooth = 1e-6 ):
        super().__init__( reduction = tf.keras.losses.Reduction.NONE)
        self.smooth = smooth
        self.alpha = alpha
        self.beta = beta
        self.weights = tf.constant([1.0 , 135.7789199161374 ,489.1624819462532 ])
        self.diceLoss = diceLoss()
        self.ce =tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False , reduction='none')

    def call(self,y_true , y_pred):
        ce=self.ce(y_true , y_pred)
        y_true = tf.squeeze(y_true,axis =-1)
        map_tensor = tf.gather(self.weights , y_true )
        ce = ce*map_tensor
        ce = tf.reduce_mean(ce , axis=[1,2,3])
        dice_loss = self.diceLoss.compute_per_sample(y_true , y_pred)
        loss = self.alpha * ce + self.beta * dice_loss
        return loss