import sys, os
import tensorflow as tf

from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy("mixed_bfloat16")

repo_path = "/content/drive/MyDrive/Medical-Segmentation-Decathlon"
sys.path.append(repo_path)

class diceLoss(tf.keras.losses.Loss):
    """
    A numerically stable, TPU-compatible Dice loss for semantic segmentation.
    """
    def __init__(self, smooth=1e-5, num_classes=4):
        super().__init__(reduction=tf.keras.losses.Reduction.NONE, name='diceLoss')
        self.smooth = smooth
        self.num_classes = num_classes

    def call(self, y_true, y_pred):
        """
        Calculates the per-sample Dice loss.
        Args:
            y_true: Ground truth labels, shape (batch, H, W, D, 1).
            y_pred: Model predictions (logits), shape (batch, H, W, D, num_classes).
        Returns:
            A tensor of shape (batch,) representing the loss for each sample.
        """

        y_pred_probs = tf.nn.softmax(y_pred, axis=-1)

        y_true_squeezed = tf.squeeze(y_true, axis=-1)
        y_true_one_hot = tf.one_hot(tf.cast(y_true_squeezed, tf.int32), depth=self.num_classes, dtype=y_pred_probs.dtype)

        intersection = tf.reduce_sum(y_true_one_hot * y_pred_probs, axis=[1, 2, 3])
        union = tf.reduce_sum(y_true_one_hot, axis=[1, 2, 3]) + tf.reduce_sum(y_pred_probs, axis=[1, 2, 3])
        
        dice_per_class = (2.0 * intersection + self.smooth) / (union + self.smooth)
        

        dice_loss_per_sample = 1.0 - tf.reduce_mean(dice_per_class, axis=-1)
        
        return dice_loss_per_sample


class Sementic_segmentation_loss(tf.keras.losses.Loss):
    """
    A combined Weighted Cross-Entropy and Dice loss for semantic segmentation.
    This loss function is hardened for TPU training with mixed precision.
    """
    def __init__(self, class_weights, loss_weights, num_classes=4):
        super().__init__(reduction=tf.keras.losses.Reduction.NONE, name='Sementic_segmentation_loss')
        self.alpha = loss_weights[0] 
        self.beta = loss_weights[1]   
        
        self.class_weights = class_weights
        #self.class_weights = tf.clip_by_value(tf.constant(class_weights, dtype=tf.float32), 1.0, 50.0)
        
        self.diceLoss = diceLoss(num_classes=num_classes)
        

        self.ce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    def call(self, y_true, y_pred):
        """
        Calculates the combined per-sample loss.
        Returns:
            A tensor of shape (batch,) representing the final loss for each sample.
        """

        y_pred_float32 = tf.cast(y_pred, tf.float32)
        y_true_float32 = tf.cast(y_true, tf.float32)
        
        y_true_squeezed = tf.squeeze(y_true_float32, axis=-1)
        

        ce_per_voxel = self.ce(y_true_squeezed, y_pred_float32)
        map_tensor = tf.gather(self.class_weights, tf.cast(y_true_squeezed, tf.int32))
        weighted_ce_per_voxel = ce_per_voxel * map_tensor
        

        ce_per_sample = tf.reduce_mean(weighted_ce_per_voxel, axis=[1, 2, 3])
        

        dice_per_sample = self.diceLoss(y_true_float32, y_pred_float32)
        

        total_loss_per_sample = (self.alpha * ce_per_sample) + (self.beta * dice_per_sample)
        
        return total_loss_per_sample