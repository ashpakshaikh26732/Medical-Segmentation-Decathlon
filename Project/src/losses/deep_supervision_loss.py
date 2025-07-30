import sys, os
import tensorflow as tf 

from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy("mixed_float16")

repo_path = "/content/drive/MyDrive/Medical-Segmentation-Decathlon"
sys.path.append(repo_path)

import tensorflow as tf

class DiceLoss3D(tf.keras.losses.Loss):

    def __init__(self, smooth=1e-6, name="DiceLoss3D"):

        super().__init__(name=name, reduction=tf.keras.losses.Reduction.NONE)
        self.smooth = smooth

    def call(self, y_true, y_pred):

        y_pred = tf.nn.softmax(y_pred, axis=-1)

        if y_true.shape.ndims == 5:
            y_true = tf.squeeze(y_true, axis=-1)

        num_classes = tf.shape(y_pred)[-1]
        y_true_one_hot = tf.one_hot(tf.cast(y_true, tf.int32), depth=num_classes, dtype=y_pred.dtype)

        intersection = tf.reduce_sum(y_true_one_hot * y_pred, axis=[1, 2, 3])
        denominator = tf.reduce_sum(y_true_one_hot, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3])

        dice_score = (2.0 * intersection + self.smooth) / (denominator + self.smooth)


        return 1.0 - tf.reduce_mean(dice_score, axis=-1)

class DeepSupervisionLoss3D(tf.keras.losses.Loss):

    def __init__(self, class_weights, output_weights=None, name="DeepSupervisionLoss3D"):

        super().__init__(name=name, reduction=tf.keras.losses.Reduction.NONE)

        self.class_weights = tf.constant(class_weights, dtype=tf.float32)
        self.output_weights = output_weights or [1.0]

        self.ce_loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
        self.dice_loss_fn = DiceLoss3D()

    def _resize_y_true_3d(self, y_true, y_pred):

        y_pred_shape = tf.shape(y_pred)
        target_h, target_w = y_pred_shape[1], y_pred_shape[2]
        y_true_reshaped = tf.transpose(y_true, perm=[0, 3, 1, 2, 4])
        y_true_reshaped = tf.reshape(y_true_reshaped, (tf.shape(y_true)[0] * tf.shape(y_true)[3], tf.shape(y_true)[1], tf.shape(y_true)[2], 1))
        y_true_resized = tf.image.resize(y_true_reshaped, size=(target_h, target_w), method='nearest')
        y_true_resized = tf.reshape(y_true_resized, (tf.shape(y_true)[0], tf.shape(y_true)[3], target_h, target_w, 1))
        y_true_resized = tf.transpose(y_true_resized, perm=[0, 2, 3, 1, 4])
        return tf.cast(y_true_resized, y_true.dtype)

    def call(self, y_true, y_preds):

        total_per_sample_loss = 0.0
        num_outputs = len(y_preds)

        final_output_weights = self.output_weights[:num_outputs] + [1.0] * (num_outputs - len(self.output_weights))

        for i in range(num_outputs):
            y_pred = y_preds[i]
            y_true_resized = self._resize_y_true_3d(y_true, y_pred)


            ce_per_voxel = self.ce_loss_fn(y_true_resized, y_pred)
            y_true_squeezed = tf.squeeze(y_true_resized, axis=-1)
            weight_map = tf.gather(self.class_weights, indices=tf.cast(y_true_squeezed, tf.int32))
            weighted_ce_per_voxel = ce_per_voxel * weight_map

            ce_per_sample_loss = tf.reduce_mean(weighted_ce_per_voxel, axis=[1, 2, 3])


            dice_per_sample_loss = self.dice_loss_fn(y_true_resized, y_pred)


            level_loss = ce_per_sample_loss + dice_per_sample_loss
            total_per_sample_loss += final_output_weights[i] * level_loss


        return total_per_sample_loss