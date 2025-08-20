import sys, os
import tensorflow as tf

from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy("mixed_float16")

repo_path = "/content/drive/MyDrive/Medical-Segmentation-Decathlon"
sys.path.append(repo_path)

from Project.src.callbacks.CheckpointCallback import *
from Project.src.callbacks.TrainingLogger import * 
from Project.src.callbacks.early_stoping import * 
from Project.src.callbacks.learning_rate_sceduler import * 



class master_callback(tf.keras.callbacks.Callback):
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
    # master_cb = master_callback(config, model, optimizer)
    # start_epoch = master_cb.on_train_begain()
    #
    # for epoch in range(start_epoch, total_epochs):
    #     master_cb.on_epoch_begain(epoch)
    #     for step, batch in enumerate(train_dataset):
    #         global_step = epoch * steps_per_epoch + step
    #         master_cb.on_batch_begain(global_step)
    #         # ... train step logic ...
    #         master_cb.on_batch_end(step, epoch, batch_logs)
    #
    #     # ... validation loop ...
    #
    #     should_stop = master_cb.on_epoch_end(epoch, epoch_logs)
    #     if should_stop:
    #         break
    """
    def __init__(self, config , model , optimizer):
        super().__init__()
        self.writer = tf.summary.create_file_writer(config['checkpoint']['log_dir'])
        self.stop = False
        self.checkpoint_dir = config['checkpoint']['checkpoint_dir']
        self._model = model
        self.config = config
        self.optimizer = optimizer
        self.target_lr = float(self.config['checkpoint']['target_lr'])
        self.warmup_step = self.config['checkpoint']['warmup_step']
        self.min_delta  = self.config['checkpoint']['min_delta']
        self.patiance = self.config['checkpoint']['patiance']
        self.batches_per_epoch = self.config['checkpoint']['batches_per_epoch']
        self.total_steps = self.config['checkpoint']['total_step']
        self.checkpoint_callback = CheckpointCallback(checkpoint_dir=self.checkpoint_dir , model=self._model , optimizer=self.optimizer)
        self.learning_rate_schduler = LearningRateScheduler(optimizer=self.optimizer, target_lr=self.target_lr,warmup_step=self.warmup_step , total_step=self.total_steps)
        self.EarlyStoppingCallback = EarlyStoppingCallback(min_delta=self.min_delta , patience=self.patiance)
        self.TrainingLogger = TrainingLogger(batches_per_epoch=self.batches_per_epoch)

    def on_train_begain(self):
        """Loads the latest checkpoint and returns the starting epoch."""
        return self.checkpoint_callback.load_latest_model()

    def on_epoch_begain(self,epoch ):
        """Delegates to the logger to signal the start of an epoch."""
        return self.TrainingLogger.on_epoch_begin(epoch=epoch)

    def on_batch_begain(self,global_step):
        """Delegates to the learning rate scheduler to update the LR."""
        return self.learning_rate_schduler.on_batch_begin(global_step=global_step)

    def on_batch_end(self, batch , epoch ,data):
        """Delegates to the logger to update batch-wise metrics."""
        return self.TrainingLogger.on_batch_end(batch=batch  , epoch = epoch , data=data)

    def on_epoch_end(self, epoch ,data):
        """Delegates logging and checkpointing, logs to TensorBoard, and checks for early stopping.

        Args:
            epoch (int): The current epoch number.
            data (dict): A dictionary containing all final logs for the epoch.

        Returns:
            bool: A flag indicating if training should stop.
        """
        self.TrainingLogger.on_epoch_end(epoch = epoch , data = data)

        per_class_iou = data['metrics']['per_class_iou']
        per_class_iou_val = data['metrics']['per_class_iou_val']
        per_class_dice = data['metrics']['per_class_dice']
        per_class_dice_val = data['metrics']['per_class_dice_val']

        with self.writer.as_default() :
            tf.summary.scalar('loss' , data['loss'], step = epoch)
            tf.summary.scalar('val_loss' , data['val_loss'] , step = epoch)

            for key, value in per_class_iou.items():
                tf.summary.scalar('iou_' + key , value , step = epoch)

            for key, value in per_class_iou_val.items():
                tf.summary.scalar('val_iou_' + key , value , step = epoch)

            for key, value in per_class_dice.items():
                tf.summary.scalar('dice_' + key , value , step = epoch)

            for key, value in per_class_dice_val.items():
                tf.summary.scalar('val_dice_'+key , value , step = epoch)

        self.writer.flush()
        self.checkpoint_callback.on_epoch_end(epoch=epoch )
        self.stop =  self.EarlyStoppingCallback.on_epoch_end(val_loss=data['val_loss'],epoch = epoch)
        return self.stop