import sys, os
import tensorflow as tf 

from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy("mixed_float16")

repo_path = "/content/drive/MyDrive/Medical-Segmentation-Decathlon"
sys.path.append(repo_path)



class EarlyStoppingCallback(tf.keras.callbacks.Callback):
    """A custom Keras Callback to stop training when validation loss stops improving.

    This callback monitors the validation loss and stops the training process
    when the loss has not improved by more than `min_delta` for a number of
    epochs equal to `patience`.

    This implementation is designed specifically for a **custom training loop**.
    The `on_epoch_end` method returns a boolean flag (`True` to stop, `False` to
    continue) which must be checked manually within the loop.

    Args:
        config (dict): A configuration dictionary. Must contain the keys
            `['data']['min_delta']` and `['data']['patience']`.
    """
    def __init__(self, min_delta,patience):
        super().__init__()
   
        self.min_delta = min_delta
        self.patience = patience
        self.best_loss = float('inf')
        self.stopped_epoch = 0
        self.wait = 0

    def on_epoch_end(self, val_loss, epoch):
        """Checks for improvement in validation loss and returns a stop signal.

        Args:
            val_loss (float): The validation loss for the current epoch.
            epoch (int): The current epoch number.

        Returns:
            bool: `True` if training should stop, `False` otherwise.
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.wait = 0
        else:
            self.wait += 1
            
        if self.wait >= self.patience:
            self.stopped_epoch = epoch
            print(f'\nEarly stopping training at epoch {self.stopped_epoch + 1}')
            return True
            
        return False