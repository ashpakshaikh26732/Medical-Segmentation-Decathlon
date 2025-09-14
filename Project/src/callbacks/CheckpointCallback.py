#import sys, os
#import tensorflow as tf 
#
#from tensorflow.keras import mixed_precision
#mixed_precision.set_global_policy("mixed_float16")
#
#repo_path = "/content/drive/MyDrive/Medical-Segmentation-Decathlon"
#sys.path.append(repo_path)
#
#import tensorflow as tf
#import os
#
#class CheckpointCallback(tf.keras.callbacks.Callback):
#    """A custom Keras Callback for robustly saving and loading training checkpoints.
#
#    This callback uses TensorFlow's `tf.train.CheckpointManager` to save the
#    complete training state, including the model weights, optimizer state,
#    and the current epoch number.
#
#    It is designed to work seamlessly with distributed training strategies
#    (like `tf.distribute.TPUStrategy`) by correctly initializing optimizer
#    slots before restoring a checkpoint. It provides a `load_latest_model`
#    method to easily resume training from the last saved state.
#
#    Args:
#        config (dict): A configuration dictionary. Must contain the key
#            `['data']['checkpoint_dir']`.
#        model (tf.keras.Model): The Keras model instance to be saved.
#        optimizer (tf.keras.optimizers.Optimizer): The optimizer instance to be saved.
#
#
#    """
#    def __init__(self, checkpoint_dir, model, optimizer):
#        super().__init__()
#
#        self.checkpoint_dir = checkpoint_dir
#        self._model = model
#        self.optimizer = optimizer
#
#        os.makedirs(self.checkpoint_dir, exist_ok=True)
#
#        self.checkpoint = tf.train.Checkpoint(
#            model=self._model,
#            optimizer=self.optimizer,
#            epoch=tf.Variable(0, dtype=tf.int32)
#        )
#
#        self.checkpoint_manager  = tf.train.CheckpointManager(
#            checkpoint=self.checkpoint,
#            directory=self.checkpoint_dir,
#            max_to_keep=5
#        )
#
#    def _initialize_optimizer_slots(self):
#        """Initializes optimizer state variables for correct checkpoint restoration."""
#        strategy = tf.distribute.get_strategy()
#        with strategy.scope():
#            self.optimizer.build(self._model.trainable_variables)
#
#    def on_epoch_end(self, epoch, logs=None):
#        """Saves a checkpoint at the end of an epoch."""
#
#        self.checkpoint.epoch.assign(epoch + 1)
#        path = self.checkpoint_manager.save()
#        print(f"\n✅ Checkpoint saved for epoch {epoch + 1} at {path}")
#
#    def load_latest_model(self):
#        """Restores the latest checkpoint and returns the epoch to resume from."""
#        latest = self.checkpoint_manager.latest_checkpoint
#        if latest:
#            print("🔧 Initializing optimizer slot variables…")
#            self._initialize_optimizer_slots()
#            print(f"📦 Restoring checkpoint from {latest}…")
#
#            self.checkpoint.restore(latest).assert_existing_objects_matched()
#            start_epoch = int(self.checkpoint.epoch.numpy())
#            
#            print(f"✅ Restored successfully. Resuming training from epoch {start_epoch + 1}.")
#            return start_epoch 
#        else:
#            print("🚀 No checkpoint found. Starting training from epoch 1.")
#            return 0



# In Project/src/callbacks/CheckpointCallback.py
import sys
import os
import tensorflow as tf
import threading
import glob

from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy("mixed_float16")

repo_path = "/content/drive/MyDrive/Medical-Segmentation-Decathlon"
sys.path.append(repo_path)

class CheckpointCallback(tf.keras.callbacks.Callback):
    """
    Final robust version of the asynchronous checkpoint callback.
    Includes a more robust restore mechanism that provides detailed error messages.
    """
    def __init__(self, local_dir, gcs_dir, model, optimizer):
        super().__init__()
        self.local_dir = local_dir
        self.gcs_dir = gcs_dir
        self._model = model
        self.optimizer = optimizer

        os.makedirs(self.local_dir, exist_ok=True)

        self.checkpoint = tf.train.Checkpoint(
            model=self._model,
            optimizer=self.optimizer,
            epoch=tf.Variable(0, dtype=tf.int32)
        )

        self.checkpoint_manager = tf.train.CheckpointManager(
            checkpoint=self.checkpoint,
            directory=self.local_dir,
            max_to_keep=5
        )

    def _upload_to_gcs_in_background(self, local_checkpoint_prefix):
        try:
            print(f"\n🚀 Starting background TF-based upload for: {os.path.basename(local_checkpoint_prefix)}")
            files_to_upload = glob.glob(f"{local_checkpoint_prefix}*")
            for local_path in files_to_upload:
                gcs_path = os.path.join(self.gcs_dir, os.path.basename(local_path))
                tf.io.gfile.copy(local_path, gcs_path, overwrite=True)

            checkpoint_tracker_file = os.path.join(self.local_dir, "checkpoint")
            if tf.io.gfile.exists(checkpoint_tracker_file):
                gcs_tracker_path = os.path.join(self.gcs_dir, "checkpoint")
                tf.io.gfile.copy(checkpoint_tracker_file, gcs_tracker_path, overwrite=True)

            print(f"\n✅ Background upload complete for: {os.path.basename(local_checkpoint_prefix)}")
        except Exception as e:
            print(f"\n🚨 ERROR in background upload thread: {e}")

    def _initialize_optimizer_slots(self):
        strategy = tf.distribute.get_strategy()
        with strategy.scope():
            self.optimizer.build(self._model.trainable_variables)

    def on_epoch_end(self, epoch, logs=None):
        self.checkpoint.epoch.assign(epoch + 1)
        local_path_prefix = self.checkpoint_manager.save()
        print(f"\n✅ Checkpoint saved locally for epoch {epoch + 1} at {local_path_prefix}")
        
        upload_thread = threading.Thread(
            target=self._upload_to_gcs_in_background,
            args=(local_path_prefix,)
        )
        upload_thread.start()

    def load_latest_model(self):
        print("\n🔄 Syncing checkpoints from GCS to local using tf.io.gfile...")
        try:
            if tf.io.gfile.exists(self.gcs_dir):
                gcs_files = tf.io.gfile.glob(os.path.join(self.gcs_dir, "*"))
                if gcs_files:
                    for gcs_path in gcs_files:
                        local_path = os.path.join(self.local_dir, os.path.basename(gcs_path))
                        tf.io.gfile.copy(gcs_path, local_path, overwrite=True)
                    print("✅ GCS sync download complete.")
        except Exception as e:
            print(f"🚨 WARNING: Could not sync from GCS. Error: {e}")


        
        tracker_file_path = os.path.join(self.local_dir, "checkpoint")
        
        if os.path.exists(tracker_file_path):
            try:

                latest_path = tf.train.latest_checkpoint(self.local_dir)
                print(f"Found latest checkpoint tracker. Attempting to restore from: {latest_path}")


                self._initialize_optimizer_slots()


                status = self.checkpoint.restore(latest_path)
                

                status.assert_existing_objects_matched()
                
                start_epoch = int(self.checkpoint.epoch.numpy())
                print(f"✅ Restored successfully. Resuming training from epoch {start_epoch + 1}.")
                return start_epoch

            except (tf.errors.NotFoundError, AssertionError) as e:

                print("\n" + "="*50)
                print("🚨 FATAL ERROR: Checkpoint found, but failed to restore.")
                print("This usually means the model architecture or optimizer has changed.")
                print(f"Underlying TensorFlow error: {e}")
                print("="*50 + "\n")
                print("Starting training from epoch 1.")
                return 0
        

        print("🚀 No 'checkpoint' tracker file found. Starting training from epoch 1.")
        return 0