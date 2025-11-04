
import sys, os
import tensorflow as tf 

from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy("mixed_float16")

repo_path = "/content/drive/MyDrive/Medical-Segmentation-Decathlon"
sys.path.append(repo_path)

class PerClassIoU(tf.keras.metrics.Metric):
    """
    Computes per-class Intersection over Union (IoU) for semantic segmentation.

    This is a stateful metric that accumulates the intersection and union totals
    across all batches in an epoch. The final IoU for each class is calculated as
    `total_intersection / total_union`.

    Args:
        n_classes (int): The total number of classes in the segmentation task.
        name (str): An optional name for the metric instance. Defaults to 'per_class_iou'.
        **kwargs: Additional arguments for the `tf.keras.metrics.Metric` parent class.

    Example:
    >>> model.compile(
    ...     optimizer='adam',
    ...     loss='...',
    ...     metrics=[PerClassIoU(n_classes=3)]
    ... )
    """
    def __init__(self, n_classes, name='per_class_iou', **kwargs):
        super().__init__(name=name, **kwargs)

        self.n_classes = n_classes
        self.smooth = 1e-6
        self.sum_intersection = self.add_weight(
            name="sum_intersection",
            shape=(self.n_classes,),
            initializer="zeros"
        )
        self.sum_union = self.add_weight(
            name="sum_union",
            shape=(self.n_classes,),
            initializer="zeros"
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        Accumulates the intersection and union statistics for a single batch.

        Args:
            y_true: The ground truth labels, of shape `(B, H, W, D, 1)`.
            y_pred: The predicted logits or probabilities, of shape `(B, H, W, D, C)`.
            sample_weight: Optional weighting of samples. Not used in this metric.
        """
        y_pred = tf.argmax(y_pred, axis=-1)

        y_pred = tf.cast(y_pred, dtype=tf.int32)
        y_true = tf.cast(y_true, dtype=tf.int32)

        if y_true.shape.ndims == 5:
            y_true = tf.squeeze(y_true, axis=-1)

        y_true_one_hot = tf.one_hot(y_true, depth=self.n_classes)
        y_pred_one_hot = tf.one_hot(y_pred, depth=self.n_classes)

        intersection = tf.reduce_sum(y_true_one_hot * y_pred_one_hot, axis=[0, 1, 2, 3])

        sum_true = tf.reduce_sum(y_true_one_hot, axis=[0, 1, 2, 3])
        sum_pred = tf.reduce_sum(y_pred_one_hot, axis=[0, 1, 2, 3])
        union = sum_true + sum_pred - intersection

        self.sum_intersection.assign_add(intersection)
        self.sum_union.assign_add(union)

    def result(self):
        """
        Computes and returns the final per-class IoU for the epoch.

        Returns:
            A tensor of shape `(n_classes,)` containing the IoU for each class.
        """
        return (self.sum_intersection + self.smooth) / (self.sum_union + self.smooth)

    def reset_state(self):
        """
        Resets all state variables to their initial values at the start of each epoch.
        """
        self.sum_intersection.assign(tf.zeros_like(self.sum_intersection))
        self.sum_union.assign(tf.zeros_like(self.sum_union))