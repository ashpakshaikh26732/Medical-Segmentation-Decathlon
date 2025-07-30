import sys, os
import tensorflow as tf

from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy("mixed_float16")

repo_path = "/content/drive/MyDrive/Medical-Segmentation-Decathlon"
sys.path.append(repo_path)

from Project.src.callbacks.CheckpointCallback import *
from Project.src.callbacksTrainingLogger import * 
from Project.src.callbacks.early_stoping import * 
from Project.src.callbacks.learning_rate_sceduler import * 

class MasterCallback(tf.keras.callbacks.Callback):
    """A master callback that composes and manages all other training callbacks.

    This callback acts as a single, centralized controller for all training
    events. It follows the composition pattern by initializing and delegating
    tasks to other specialized callbacks for checkpointing, learning rate
    scheduling, early stopping, and interactive logging.

    It also includes logic for logging all metrics to TensorBoard at the end
    of each epoch.

    This implementation is designed specifically for a **custom training loop**
    that manually calls the `on_*` methods at the appropriate times.

    Args:
        config (dict): A configuration dictionary containing all parameters
            required by the composed callbacks (e.g., checkpoint_dir, lr_params,
            early_stopping_params, etc.).
        model (tf.keras.Model): The Keras model instance.
        optimizer (tf.keras.optimizers.Optimizer): The Keras optimizer instance.
    
    Example:
    # In your custom training loop:
    # master_cb = MasterCallback(config, model, optimizer)
    # start_epoch = master_cb.on_train_begin()
    #
    # for epoch in range(start_epoch, total_epochs):
    #     master_cb.on_epoch_begin(epoch)
    #     for step, batch in enumerate(train_dataset):
    #         global_step = epoch * steps_per_epoch + step
    #         master_cb.on_batch_begin(global_step)
    #         # ... train step logic ...
    #         master_cb.on_batch_end(step, epoch, batch_logs)
    #
    #     # ... validation loop ...
    #
    #     should_stop = master_cb.on_epoch_end(epoch, epoch_logs)
    #     if should_stop:
    #         break
    """
    def __init__(self, config, model, optimizer):
        super().__init__()
        self.config = config
        self.model = model
        self.optimizer = optimizer
        

        self.checkpoint_callback = CheckpointCallback(
            config=self.config, 
            model=self.model, 
            optimizer=self.optimizer
        )
        self.learning_rate_scheduler = LearningRateScheduler(
            config=self.config, 
            optimizer=self.optimizer
        )
        self.early_stopping_callback = EarlyStoppingCallback(config=self.config)
        self.training_logger = TrainingLogger(config=self.config)


        log_dir = self.config['data'].get('log_dir', 'logs/')
        self.writer = tf.summary.create_file_writer(log_dir)
        self.stop = False

    def on_train_begin(self, logs=None):
        """Loads the latest checkpoint and returns the starting epoch."""
        return self.checkpoint_callback.load_latest_model()
    
    def on_epoch_begin(self, epoch, logs=None):
        """Delegates to the logger to signal the start of an epoch."""
        self.training_logger.on_epoch_begin(epoch=epoch)

    def on_batch_begin(self, global_step, logs=None):
        """Delegates to the learning rate scheduler to update the LR."""
        self.learning_rate_scheduler.on_batch_begin(global_step=global_step)
    
    def on_batch_end(self, batch, epoch, data):
        """Delegates to the logger to update batch-wise metrics."""
        self.training_logger.on_batch_end(batch=batch, epoch=epoch, data=data)
    
    def on_epoch_end(self, epoch, data):
        """Delegates logging and checkpointing, logs to TensorBoard, and checks for early stopping.
        
        Args:
            epoch (int): The current epoch number.
            data (dict): A dictionary containing all final logs for the epoch.

        Returns:
            bool: A flag indicating if training should stop.
        """

        self.training_logger.on_epoch_end(epoch=epoch, data=data)


        with self.writer.as_default():
            tf.summary.scalar('loss', data['loss'], step=epoch)
            tf.summary.scalar('val_loss', data['val_loss'], step=epoch)

            for key, value in data['metrics']['per_class_iou'].items():
                tf.summary.scalar(f'iou/{key}', value, step=epoch)
            for key, value in data['metrics']['per_class_iou_val'].items():
                tf.summary.scalar(f'val_iou/{key}', value, step=epoch)
            for key, value in data['metrics']['per_class_dice'].items():
                tf.summary.scalar(f'dice/{key}', value, step=epoch)
            for key, value in data['metrics']['per_class_dice_val'].items():
                tf.summary.scalar(f'val_dice/{key}', value, step=epoch)
        self.writer.flush()
        

        self.checkpoint_callback.on_epoch_end(epoch=epoch)
        

        self.stop = self.early_stopping_callback.on_epoch_end(
            val_loss=data['val_loss'], 
            epoch=epoch
        )
        return self.stop