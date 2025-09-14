import sys, os
import tensorflow as tf 

from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy("mixed_float16")

repo_path = "/content/drive/MyDrive/Medical-Segmentation-Decathlon"
sys.path.append(repo_path)


import pandas as pd
import tensorflow as tf
from tqdm.notebook import tqdm
from IPython.display import display, HTML
import matplotlib.pyplot as plt
import numpy as np

#class TrainingLogger(tf.keras.callbacks.Callback):
#    """A custom Keras Callback for rich, interactive logging in notebook environments.
#
#    This callback provides a detailed, live-updating log during model training,
#    making it ideal for use in Jupyter or Google Colab. It is designed to be
#    used with a custom training loop that manually passes log data.
#
#    Features:
#    - A `tqdm` progress bar to track batch-wise progress within an epoch.
#    - A `pandas` DataFrame that updates in-place after each batch with the
#      latest loss and per-class metrics.
#    - A summary table at the end of each epoch with final training and
#      validation metrics.
#    - A multi-panel `matplotlib` plot at the end of each epoch visualizing the
#      trends of batch loss, per-class IoU, and per-class Dice scores.
#
#    Args:
#        config (dict): A configuration dictionary. Must contain the key
#            `['data']['batches_per_epoch']` to set up the progress bar.
#
#    Note on Custom Training Loops:
#        The `on_batch_end` and `on_epoch_end` methods expect a custom `data`
#        dictionary to be passed from the training loop. The expected structure is:
#
#        For `on_batch_end(..., data=data)`:
#        data = {
#            'loss': 0.123,
#            'metrics': {
#                'per_class_iou': {'class_0': 0.9, 'class_1': 0.8},
#                'per_class_dice': {'class_0': 0.95, 'class_1': 0.88}
#            }
#        }
#
#        For `on_epoch_end(..., data=data)`:
#        data = {
#            'loss': 0.100,
#            'val_loss': 0.150,
#            'metrics': {
#                'per_class_iou': {...},
#                'per_class_iou_val': {...},
#                'per_class_dice': {...},
#                'per_class_dice_val': {...}
#            }
#        }
#    """
#    def __init__(self, batches_per_epoch):
#        super().__init__()
#        self.batches_per_epoch = batches_per_epoch
#        self.display_table = None
#        self.graph_display = None
#
#    def on_epoch_begin(self, epoch, logs=None):
#        """Initializes progress bar and data stores at the start of an epoch."""
#        print(f'Epoch {epoch + 1} Begins')
#        self.progress_bar = tqdm(total=self.batches_per_epoch, desc=f'Epoch {epoch + 1}')
#        self.batch_table_data = []
#        self.batch_loss = []
#        self.batch_per_class_iou = []
#        self.batch_per_class_dice = []
#        self.epoch_table_data = []
#        display(HTML('<h3>📊 Batch-wise Metrics</h3>'))
#        # Initialize display for batch table
#        self.display_table = display(pd.DataFrame(self.batch_table_data), display_id=True)
#
#
#    def on_batch_end(self, batch, epoch, data=None):
#        """Updates progress bar and live table after each batch."""
#        self.progress_bar.set_postfix({'loss': data.get('loss', 0.0)})
#        self.progress_bar.update(1)
#
#        if not data:
#            return
#            
#        per_class_iou = data['metrics']['per_class_iou']
#        per_class_dice = data['metrics']['per_class_dice']
#
#        batch_dict = {
#            'batch': batch + 1,
#            'loss': data['loss']
#        }
#        for key, value in per_class_iou.items():
#            batch_dict['class_iou_' + key] = value
#        for key, value in per_class_dice.items():
#            batch_dict['class_dice_' + key] = value
#        
#        self.batch_table_data.append(batch_dict)
#        df = pd.DataFrame(self.batch_table_data)
#        
#        if self.display_table is None : 
#            display(HTML('📉 Loss Graphs (Batch-wise Trend)'))
#            self.display_table = display(df , display_id=True)
#        else :  
#            self.display_table.update(df)
#            
#        self.batch_loss.append(data['loss'])
#        self.batch_per_class_iou.append(list(per_class_iou.values()))
#        self.batch_per_class_dice.append(list(per_class_dice.values()))
#
#    def on_epoch_end(self, epoch, data=None):
#        """Displays summary table and plots at the end of an epoch."""
#        self.progress_bar.close()
#        if data is None:
#            return
#            
#        epoch_dict = {
#            'epoch': epoch + 1,
#            'loss': data['loss'],
#            'val_loss': data['val_loss']
#        }
#        per_class_iou = data['metrics']['per_class_iou']
#        per_class_iou_val = data['metrics']['per_class_iou_val']
#        per_class_dice = data['metrics']['per_class_dice']
#        per_class_dice_val = data['metrics']['per_class_dice_val']
#
#        for key, value in per_class_iou.items():
#            epoch_dict['iou_' + key] = value
#        for key, value in per_class_dice.items():
#            epoch_dict['dice_' + key] = value
#        for key, value in per_class_iou_val.items():
#            epoch_dict['val_iou_' + key] = value
#        for key, value in per_class_dice_val.items():
#            epoch_dict['val_dice_' + key] = value
#        
#        display(HTML('<h3>📊 Epoch Summary</h3>'))
#        df = pd.DataFrame([epoch_dict])
#        display(df)
#
#        fig, axes = plt.subplots(1, 3, figsize=(20, 5))
#        
#        axes[0].plot(self.batch_loss, color='red', marker='o', markersize=4)
#        axes[0].set_title(f'Epoch {epoch + 1}: Training Loss')
#        axes[0].set_xlabel('Batch')
#        axes[0].set_ylabel('Loss')
#        axes[0].grid(True)
#        
#        class_names = list(per_class_iou.keys())
#        
#        iou_data = np.array(self.batch_per_class_iou)
#        for j, key in enumerate(class_names):
#            axes[1].plot(iou_data[:, j], label=key, marker='o', markersize=4)
#        axes[1].set_title('Per-Class IoU')
#        axes[1].set_xlabel('Batch')
#        axes[1].set_ylabel('IoU')
#        axes[1].legend()
#        axes[1].grid(True)
#
#        dice_data = np.array(self.batch_per_class_dice)
#        for j, key in enumerate(class_names):
#            axes[2].plot(dice_data[:, j], label=key, marker='o', markersize=4)
#        axes[2].set_title('Per-Class Dice')
#        axes[2].set_xlabel('Batch')
#        axes[2].set_ylabel('Dice Score')
#        axes[2].legend()
#        axes[2].grid(True)
#
#        plt.tight_layout()
#        display(HTML(f'<h3>📉 Final Graphs - Epoch {epoch + 1}</h3>'))
#        display(fig)
#        plt.close(fig)


class TrainingLogger(tf.keras.callbacks.Callback):
    """
    A lightweight, performance-optimized Keras Callback for notebook training.

    This version is optimized to minimize CPU overhead by:
    - Maintaining the essential `tqdm` progress bar for real-time feedback.
    - Removing all high-frequency, per-batch DataFrame updates and HTML re-rendering.
    - Removing all expensive end-of-epoch matplotlib graph generation.
    - Displaying only a single, lightweight summary table at the end of each epoch.
    """
    def __init__(self, batches_per_epoch):
        super().__init__()
        self.batches_per_epoch = batches_per_epoch
        self.progress_bar = None

    def on_epoch_begin(self, epoch, logs=None):
        """Initializes the progress bar at the start of an epoch."""
        print(f'Epoch {epoch + 1} Begins')
        self.progress_bar = tqdm(total=self.batches_per_epoch, desc=f'Epoch {epoch + 1}')

    def on_batch_end(self, batch, epoch, data=None):
        """Updates only the lightweight progress bar after each batch."""
        # Set the postfix with the current loss. This is a very low-overhead operation.
        self.progress_bar.set_postfix({'loss': data.get('loss', 0.0)})
        self.progress_bar.update(1)

    def on_epoch_end(self, epoch, data=None):
        """Closes the progress bar and displays a final, simple summary table."""
        if self.progress_bar:
            self.progress_bar.close()

        if data is None:
            return

        # Create and display a single, simple summary DataFrame at the very end.
        # This is a low-frequency, acceptable operation.
        epoch_dict = {
            'epoch': epoch + 1,
            'loss': f"{data['loss']:.4f}",
            'val_loss': f"{data['val_loss']:.4f}"
        }
        
        # Add final validation metrics to the summary
        val_dice = data['metrics']['per_class_dice_val']
        for key, value in val_dice.items():
            epoch_dict[f'val_dice_{key}'] = f"{value:.4f}"

        display(HTML('<h3>📊 Epoch Summary</h3>'))
        df = pd.DataFrame([epoch_dict])
        display(df)