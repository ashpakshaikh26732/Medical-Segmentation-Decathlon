import sys, os
import tensorflow as tf

from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy("mixed_bfloat16")

repo_path = "/content/drive/MyDrive/Medical-Segmentation-Decathlon"
sys.path.append(repo_path)

import tensorflow as tf

class DiceLoss3D(tf.keras.losses.Loss):
    """
    A numerically stable, graph-compatible Dice loss for 3D semantic segmentation.
    This version is hardened against all sources of NaN and graph compilation errors.
    """
    def __init__(self, smooth=1e-5, name="DiceLoss3D"):
        super().__init__(name=name, reduction=tf.keras.losses.Reduction.NONE)
        self.smooth = smooth

    def call(self, y_true, y_pred):

        y_pred = tf.cast(y_pred, tf.float32)
        y_true = tf.cast(y_true, tf.float32)
        

        probs = tf.nn.softmax(y_pred, axis=-1)
        

        y_true_squeezed = tf.squeeze(y_true, axis=-1)


        num_classes = tf.shape(probs)[-1]
        y_true_one_hot = tf.one_hot(tf.cast(y_true_squeezed, tf.int32), depth=num_classes, dtype=probs.dtype)

        intersection = tf.reduce_sum(y_true_one_hot * probs, axis=[1, 2, 3])
        sum_true = tf.reduce_sum(y_true_one_hot, axis=[1, 2, 3])
        sum_pred = tf.reduce_sum(probs, axis=[1, 2, 3])

        epsilon = tf.keras.backend.epsilon()

        numerator = 2.0 * intersection + epsilon
        denominator = sum_true + sum_pred + epsilon
        
        dice_score = numerator / denominator
        
        return 1.0 - tf.reduce_mean(dice_score, axis=-1)

class DeepSupervisionLoss3D(tf.keras.losses.Loss):
    """
    Calculates a combined loss from multiple output heads of a deeply-supervised model.
    This version is hardened to be fully graph-compatible and numerically stable.
    """
    def __init__(self, class_weights, output_weights , loss_weights, name="DeepSupervisionLoss3D"):
        super().__init__(name=name, reduction=tf.keras.losses.Reduction.NONE)


        self.output_weights = output_weights 
        self.alpha = loss_weights[0]
        self.beta = loss_weights[1]
        self.class_weights = class_weights
        self.ce_loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
        self.dice_loss_fn = DiceLoss3D()

    def _resize_y_true_3d(self, y_true, y_pred):

        y_true_shape = tf.shape(y_true)
        y_pred_shape = tf.shape(y_pred)
        condition = tf.reduce_all(y_true_shape[1:4] == y_pred_shape[1:4])

        def true_fn():
            return tf.cast(y_true, y_pred.dtype)

        def false_fn():
            target_D, target_H, target_W = y_pred_shape[1], y_pred_shape[2], y_pred_shape[3]
            B, D, H, W, C = y_true_shape[0], y_true_shape[1], y_true_shape[2], y_true_shape[3], y_true_shape[4]
            y_true_float = tf.cast(y_true, tf.float32)
            

            reshaped_hw = tf.reshape(y_true_float, [B * D, H, W, C])
            resized_hw = tf.image.resize(reshaped_hw, [target_H, target_W], method='nearest')
            y_true_resized_hw = tf.reshape(resized_hw, [B, D, target_H, target_W, C])

 
            transposed_d = tf.transpose(y_true_resized_hw, [0, 2, 3, 1, 4])
            reshaped_d = tf.reshape(transposed_d, [B * target_H * target_W, D, C, 1])
            resized_d = tf.image.resize(reshaped_d, [target_D, C], method='nearest')
            reshaped_back_d = tf.reshape(resized_d, [B, target_H, target_W, target_D, C])
            
            final_resized = tf.transpose(reshaped_back_d, [0, 3, 1, 2, 4])
            return tf.cast(final_resized, y_pred.dtype)

        return tf.cond(condition, true_fn, false_fn)

    def call(self, y_true, y_preds):

        all_level_losses = []

        if not isinstance(y_preds, (list, tuple)):
            y_preds = [y_preds]

        num_outputs = len(y_preds)
        final_output_weights = self.output_weights[:num_outputs] + [1.0] * (num_outputs - len(self.output_weights))

        for i in range(num_outputs):
            y_pred = y_preds[i]
            y_pred = tf.cast(y_pred , dtype = tf.float32)
            y_true = tf.cast(y_true , dtype = tf.float32)
            
    
            logits = tf.cast(y_pred, tf.float32)
            y_true_resized = self._resize_y_true_3d(y_true, logits)
            y_true_squeezed = tf.squeeze(y_true_resized, axis=-1)


            ce_per_voxel = self.ce_loss_fn(y_true_squeezed, logits)
            weight_map = tf.gather(self.class_weights, indices=tf.cast(y_true_squeezed, tf.int32))
            weighted_ce = ce_per_voxel * weight_map
            ce_per_sample_loss = tf.reduce_mean(weighted_ce, axis=[1, 2, 3])
            
  
            dice_per_sample_loss = self.dice_loss_fn(y_true_resized, logits)

        
            level_loss = self.alpha * tf.cast(ce_per_sample_loss, tf.float32) +self.beta * tf.cast(dice_per_sample_loss, tf.float32)
            
       
            all_level_losses.append(final_output_weights[i] * level_loss)
            

        total_per_sample_loss = tf.add_n(all_level_losses)
 
            
        return total_per_sample_loss