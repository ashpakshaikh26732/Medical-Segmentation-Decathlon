import sys, os
import tensorflow as tf 

from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy("mixed_float16")

repo_path = "/content/drive/MyDrive/Medical-Segmentation-Decathlon"
sys.path.append(repo_path)

import tensorflow as tf
import os

class CheckpointCallback(tf.keras.callbacks.Callback):
    """A custom Keras Callback for robustly saving and loading training checkpoints.

    This callback uses TensorFlow's `tf.train.CheckpointManager` to save the
    complete training state, including the model weights, optimizer state,
    and the current epoch number.

    It is designed to work seamlessly with distributed training strategies
    (like `tf.distribute.TPUStrategy`) by correctly initializing optimizer
    slots before restoring a checkpoint. It provides a `load_latest_model`
    method to easily resume training from the last saved state.

    Args:
        config (dict): A configuration dictionary. Must contain the key
            `['data']['checkpoint_dir']`.
        model (tf.keras.Model): The Keras model instance to be saved.
        optimizer (tf.keras.optimizers.Optimizer): The optimizer instance to be saved.


    """
    def __init__(self, checkpoint_dir, model, optimizer):
        super().__init__()

        self.checkpoint_dir = checkpoint_dir
        self.model = model
        self.optimizer = optimizer

        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.checkpoint = tf.train.Checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            epoch=tf.Variable(0, dtype=tf.int32)
        )

        self.checkpoint_manager  = tf.train.CheckpointManager(
            checkpoint=self.checkpoint,
            directory=self.checkpoint_dir,
            max_to_keep=5
        )

    def _initialize_optimizer_slots(self):
        """Initializes optimizer state variables for correct checkpoint restoration."""
        strategy = tf.distribute.get_strategy()
        with strategy.scope():
            self.optimizer.build(self.model.trainable_variables)

    def on_epoch_end(self, epoch, logs=None):
        """Saves a checkpoint at the end of an epoch."""

        self.checkpoint.epoch.assign(epoch + 1)
        path = self.checkpoint_manager.save()
        print(f"\n✅ Checkpoint saved for epoch {epoch + 1} at {path}")

    def load_latest_model(self):
        """Restores the latest checkpoint and returns the epoch to resume from."""
        latest = self.checkpoint_manager.latest_checkpoint
        if latest:
            print("🔧 Initializing optimizer slot variables…")
            self._initialize_optimizer_slots()
            print(f"📦 Restoring checkpoint from {latest}…")

            self.checkpoint.restore(latest).assert_existing_objects_matched()
            start_epoch = int(self.checkpoint.epoch.numpy())
            
            print(f"✅ Restored successfully. Resuming training from epoch {start_epoch + 1}.")
            return start_epoch 
        else:
            print("🚀 No checkpoint found. Starting training from epoch 1.")
            return 0