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
# # In Project/src/callbacks/CheckpointCallback.py
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
# # In Project/src/callbacks/CheckpointCallback.py
import os
import sys
import threading
import glob
import shutil
import re
import tensorflow as tf
from huggingface_hub import HfApi, snapshot_download, CommitOperationDelete



# from tensorflow.keras import mixed_precision
# mixed_precision.set_global_policy("mixed_float16")

class CheckpointCallback(tf.keras.callbacks.Callback):
    
    def __init__(self, local_dir, hf_repo_id, hf_path_in_repo, model, optimizer,duration_to_save_checkpoints , remote_max_to_keep=5 ):
        super().__init__()
        self.local_dir = local_dir
        self.hf_repo_id = hf_repo_id
        self.hf_path_in_repo = hf_path_in_repo
        self._model = model
        self.optimizer = optimizer
        self.remote_max_to_keep = remote_max_to_keep
        self.duration_to_save_checkpoints = duration_to_save_checkpoints
        
        HF_TOKEN=os.getenv("HF_TOKEN")
        # This automatically picks up HF_TOKEN from environment
        self.api = HfApi(token =HF_TOKEN)
        
        os.makedirs(self.local_dir, exist_ok=True)

        self.checkpoint = tf.train.Checkpoint(
            model=self._model,
            optimizer=self.optimizer,
            epoch=tf.Variable(0, dtype=tf.int32)
        )
        self.checkpoint_manager = tf.train.CheckpointManager(
            checkpoint=self.checkpoint,
            directory=self.local_dir,
            max_to_keep=5 # Keep 5 locally for quick access
        )

    def _cleanup_old_remote_checkpoints(self):
        """Deletes old checkpoints from HF Hub to save space."""
        try:
            # 1. List all files in the remote repo
            remote_files = self.api.list_repo_files(repo_id=self.hf_repo_id)
            # Filter for files only in our specific subfolder
            my_files = [f for f in remote_files if f.startswith(self.hf_path_in_repo)]

            # 2. Find unique checkpoint numbers (e.g., extracts '15' from 'task1/ckpt-15.index')
            ckpt_numbers = set()
            for f in my_files:
                match = re.search(r'ckpt-(\d+)', f)
                if match:
                    ckpt_numbers.add(int(match.group(1)))
            
            sorted_ckpts = sorted(list(ckpt_numbers))

            # 3. If we have more than the limit, delete the oldest ones
            if len(sorted_ckpts) > self.remote_max_to_keep:
                to_delete_nums = sorted_ckpts[:-self.remote_max_to_keep]
                print(f"\n🧹 [Cleanup] Deleting old epochs from HF: {to_delete_nums}")

                # Find all files associated with these old numbers
                operations = []
                for f in my_files:
                    for num in to_delete_nums:
                        # Match 'ckpt-N.' exactly to avoid partial matches
                        if f"ckpt-{num}." in f:
                            operations.append(CommitOperationDelete(path_in_repo=f))
                            break 
                
                if operations:
                    self.api.create_commit(
                        repo_id=self.hf_repo_id,
                        operations=operations,
                        commit_message=f"Auto-cleanup: deleting epochs {to_delete_nums}",
                        repo_type="model"
                    )
                    print("✅ [Cleanup] Finished cleaning HF.")
        except Exception as e:
            print(f"⚠️ [Cleanup Error] Failed to clean old checkpoints: {e}")

    def _background_upload_task(self, local_path_prefix, epoch):
        """Background thread: Uploads new files, THEN cleans old ones."""
        try:
            # --- STEP 1: UPLOAD ---
            commit_message = f"Upload checkpoints for {self.hf_path_in_repo} - epoch {epoch}"
            # print(f"\n🚀 Starting background HF upload for epoch {epoch}...")

            files_to_upload = glob.glob(f"{local_path_prefix}*")
            tracker_file = os.path.join(self.local_dir, "checkpoint")
            if os.path.exists(tracker_file):
                files_to_upload.append(tracker_file)
            
            for local_file_path in files_to_upload:
                repo_file_path = os.path.join(self.hf_path_in_repo, os.path.basename(local_file_path))
                self.api.upload_file(
                    path_or_fileobj=local_file_path,
                    path_in_repo=repo_file_path,
                    repo_id=self.hf_repo_id,
                    repo_type="model",
                    commit_message=commit_message,
                )
            print(f"✅ [HF] Epoch {epoch} uploaded.")

            # --- STEP 2: CLEANUP ---
            # Run cleanup every time (or change to 'if epoch % 5 == 0:' to run less often)
            self._cleanup_old_remote_checkpoints()

        except Exception as e:
            print(f"\n🚨 ERROR in background HF task: {e}")

    def on_epoch_end(self, epoch, logs=None):
        if (epoch+1) % self.duration_to_save_checkpoints  : 
            self.checkpoint.epoch.assign(epoch + 1)

            local_path_prefix = self.checkpoint_manager.save()
            print(f"\n💾 Saved locally to {local_path_prefix}")
        
        # Start background upload & clean

            upload_thread = threading.Thread(
                target=self._background_upload_task,
                args=(local_path_prefix, epoch + 1) 
            )
            upload_thread.start()

    def _initialize_optimizer_slots(self):
        strategy = tf.distribute.get_strategy()
        with strategy.scope():
            self.optimizer.build(self._model.trainable_variables)

    def load_latest_model(self):
        print(f"\n🔄 Syncing from HF: {self.hf_repo_id}/{self.hf_path_in_repo}...")
        try:
            snapshot_download(
                repo_id=self.hf_repo_id,
                repo_type="model",
                local_dir=self.local_dir,
                allow_patterns=f"{self.hf_path_in_repo}/*", 
                local_dir_use_symlinks=False
            )
            print("✅ Download complete. Flattening directory structure...")


            for root, dirs, files in os.walk(self.local_dir):
                if root == self.local_dir:
                    continue 
                for file in files:
                    src_path = os.path.join(root, file)
                    dst_path = os.path.join(self.local_dir, file)
                    if not os.path.exists(dst_path):
                        shutil.move(src_path, dst_path)
            # -----------------------------

        except Exception as e:
            print(f"ℹ️ Could not sync from HF (normal for first run or network issues).")

  
        latest_checkpoint = tf.train.latest_checkpoint(self.local_dir)

        if latest_checkpoint:
            print(f"📦 Found checkpoint: {latest_checkpoint}")
            try:
                self._initialize_optimizer_slots()
                self.checkpoint.restore(latest_checkpoint).assert_existing_objects_matched()
                print(f"✅ Restored! Resuming from epoch {int(self.checkpoint.epoch.numpy()) + 1}.")
                return int(self.checkpoint.epoch.numpy())
            except Exception as e:
                print(f"🚨 Checkpoint found but failed to load: {e}")
        else:
            print(f"⚠️ No 'checkpoint' file found in {self.local_dir}. Starting from Epoch 1.")
        
        return 0
