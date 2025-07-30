import tensorflow as tf
import math

import sys, os

from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy("mixed_float16")

repo_path = "/content/drive/MyDrive/Medical-Segmentation-Decathlon"
sys.path.append(repo_path)

class LearningRateScheduler(tf.keras.callbacks.Callback):
    """A custom Keras Callback for a Cosine Decay with Warmup learning rate schedule.

    This callback adjusts the learning rate at the beginning of each batch
    following a two-phase schedule:
    1.  **Warmup Phase:** Linearly increases the learning rate from 0 to the
        `target_lr` over the initial `warmup_steps`.
    2.  **Cosine Decay Phase:** Smoothly decreases the learning rate from
        `target_lr` down to near 0 over the remaining steps, following a
        cosine curve.

    This implementation is designed specifically for a **custom training loop**
    that manually calls the `on_batch_begin` method with the global step count.

    Args:
        config (dict): A configuration dictionary. Must contain the keys
            `['data']['total_step']`, `['data']['target_lr']`, and
            `['data']['warmup_step']`.
        optimizer (tf.keras.optimizers.Optimizer): The optimizer instance whose
            learning rate will be updated.
    

    """
    def __init__(self, config, optimizer):
        super().__init__()
        self.total_steps = config['data']['total_step']
        self.target_lr = config['data']['target_lr']
        self.warmup_steps = config['data']['warmup_step']
        self.optimizer = optimizer
        self.pi = tf.constant(math.pi, dtype=tf.float32)

    def on_batch_begin(self, global_step, logs=None):
        """Calculates and sets the learning rate for the current global step.
        
        Args:
            global_step (int): The total number of batches processed since the
                beginning of training.
            logs (dict): Optional dict of logs. Not used in this implementation.
        """
        global_step = tf.cast(global_step, dtype=tf.float32)
        
        if global_step < self.warmup_steps:

            lr = self.target_lr * (global_step / self.warmup_steps)
        else:

            lr = self.target_lr * 0.5 * (1 + tf.cos(self.pi * ((global_step - self.warmup_steps) / (self.total_steps - self.warmup_steps))))


        tf.keras.backend.set_value(self.optimizer.learning_rate, lr)