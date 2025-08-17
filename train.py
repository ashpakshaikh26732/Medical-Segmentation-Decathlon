"""
A script to train 3D medical image segmentation models on the Medical
Segmentation Decathlon dataset using a custom, distributed training loop on TPUs.

This script serves as the main entry point for the training pipeline. It is a
configuration-driven script that handles everything from data downloading and
preprocessing to model instantiation, training, and logging.

It is designed to be run from the command line and uses a YAML file to
configure all aspects of the training run, including the model architecture,
data paths, hyperparameters, and callback settings. The training is performed
using a custom loop optimized for `tf.distribute.TPUStrategy`.

Args:
    --config (str): Required. The path to the YAML configuration file that
                    defines the experiment.

YAML Configuration:
    The YAML configuration file controls the entire training process. It is
    expected to contain the following main sections:

    - `model`: Specifies the model architecture to use (e.g., 'unet_plus_plus').
    - `data`: Contains all parameters for the DataPipeline, such as paths,
      number of classes, batch size, patch shape, and class names.
    - `optimizer`: Defines optimizer settings like the learning rate and weight decay.
    - `loss`: Specifies which loss function to use from the loss registry.
    - `checkpoint`: Contains all parameters for the callbacks, including the
      checkpoint directory, learning rate schedule parameters, early stopping
      patience, and logging settings.

Workflow:
    1. Parses the `--config` argument.
    2. Loads the specified YAML file.
    3. Initializes the TPU strategy.
    4. Downloads and extracts the dataset.
    5. Instantiates the DataPipeline, model, loss, optimizer, metrics, and
       the master callback.
    6. Enters a custom training loop, calling the master callback's hooks at
       each stage (train begin, epoch begin, batch begin/end, epoch end).
    7. The loop includes distributed train and validation steps, metric updates,
       and logging.
    8. Training continues until the specified number of epochs is reached or
       early stopping is triggered by the master callback.
"""
import sys, os
import tensorflow as tf



from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy("mixed_float16")

repo_path = "/content/drive/MyDrive/Medical-Segmentation-Decathlon"
sys.path.append(repo_path)

from Project.src.callbacks.master_callback import *
from Project.src.data.pipeline import *
from Project.src.losses.deep_supervision_loss import *
from Project.src.losses.sementic_segmentation_loss import *
from Project.src.metrics.Per_class_dice import *
from Project.src.metrics.per_class_iou import *
from Project.src.models.swinunet import *
from Project.src.models.transunet import *
from Project.src.models.unetpp import *

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import yaml
import requests
import tarfile

def download_and_extract(url, output_dir=".", tarfile_name="dataset.tar"):
    """Downloads and extracts a TAR file without using MONAI."""
    tar_path = os.path.join(output_dir, tarfile_name)
    
    print(f"Downloading dataset from {url}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    with open(tar_path, 'wb') as f, tqdm(
        desc=tarfile_name,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            size = f.write(chunk)
            bar.update(size)
            
    print(f"\nExtracting {tarfile_name}...")
    with tarfile.open(tar_path) as tar:
        tar.extractall(path=output_dir)
    print("Extraction complete.")
    os.remove(tar_path) 

parser = argparse.ArgumentParser(description='Deep learning Training Script')
parser.add_argument('--config',type=str , required=True , help='path to yaml file')
args = parser.parse_args()

with open(args.config ,'r') as f :
    config = yaml.safe_load(f)

try:

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='local')
    

    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    

    strategy = tf.distribute.TPUStrategy(tpu)
    
    print("✅ TPU is running on:", tpu.master())

except ValueError as e:
    print("❌ TPU is not available:", e)

model_registry = {
    "unet_plus_plus" :UNET_PLUS_PLUS ,
    'swimUnet' : SwimUnet ,
    'transUnet' : TRANSUNET
}
loss_registry = {
'sementic_segmetation_loss' : Sementic_segmentation_loss,
'deep_supervision_loss' : DeepSupervisionLoss3D
}

download_and_extract(config['data']['dataset_url'], output_dir="/kaggle/working/", tarfile_name=config['data']['tarfile_name'])
task_name = os.path.splitext(config['data']['tarfile_name'])[0]
image_address  = os.path.join('/kaggle/working', task_name , 'imagesTr')
label_address = os.path.join('/kaggle/working',task_name , 'labelsTr')
dataPipeline= DataPipeline(config , image_address , label_address)
train_dataset , val_dataset = dataPipeline.load_for_preprocessing()
train_dataset = strategy.experimental_distribute_dataset(train_dataset)
val_dataset = strategy.experimental_distribute_dataset(val_dataset)



with strategy.scope() :
    model = model_registry[config['model']['name']]()
    sample_input = tf.random.uniform(shape=tuple(config['data']['patch_shape']))
    if config['model']['name'] =='unet_plus_plus':
        loss_fn  = loss_registry[config['loss']](config['data']['class_weights'])
    else :
        loss_fn = loss_registry[config['loss']]()
    model(sample_input)
    print("Model built:", model.built)
    optimizer = tf.keras.optimizers.AdamW(learning_rate=config['optimizer']['starting_lr'],weight_decay=config['optimizer']['weight_decay'])
    per_class_iou = PerClassIoU(config['data']['num_classes'])
    per_class_dice = PerClassDice(config['data']['num_classes'])
    per_class_iou_val = PerClassIoU(config['data']['num_classes'])
    per_class_dice_val = PerClassDice(config['data']['num_classes'])

master_callback=master_callback(config , model , optimizer)

with strategy.scope():

    def train_step(dist_input):
        x_train , y_train=dist_input
        if config['model']['name'] == 'unet_plus_plus':
            with tf.GradientTape() as tape :
                model_output=model(x_train)
                per_example_loss = loss_fn(y_train , model_output)
                model_logits = model_output[-1]
                loss =  tf.nn.compute_average_loss(per_example_loss=per_example_loss , global_batch_size=dataPipeline.final_batch_size)
        else :
            with tf.GradientTape() as tape :
                model_logits = model(x_train)
                per_example_loss = loss_fn(y_train , model_logits)
                loss = tf.nn.compute_average_loss(per_example_loss=per_example_loss , global_batch_size=dataPipeline.final_batch_size)

        gradients = tape.gradient(loss , model.trainable_variables)
        optimizer.apply_gradients(zip(gradients , model.trainable_variables))

        per_class_iou.update_state(y_true = y_train , y_pred = model_logits )
        per_class_dice.update_state(y_true = y_train , y_pred = model_logits)

        return loss

    def val_step (dist_input):
        x_val , y_val = dist_input
        if config['model']['name'] == 'unet_plus_plus':
            model_output = model(x_val)
            model_logits = model_output[-1]
            per_example_loss = loss_fn(y_val , model_output)
            val_loss = tf.nn.compute_average_loss(per_example_loss=per_example_loss , global_batch_size=dataPipeline.final_batch_size)
        else :
            model_logits = model(x_val)
            per_example_loss = loss_fn(y_val , model_logits)
            val_loss  = tf.nn.compute_average_loss(per_example_loss=per_example_loss , global_batch_size=dataPipeline.final_batch_size)

        per_class_iou_val.update_state(y_val , model_logits)
        per_class_dice_val.update_state(y_val , model_logits)

        return val_loss

    @tf.function
    def distributed_train_step(dist_data):
        per_replica_loss = strategy.run(train_step , args=(dist_data, ))
        return strategy.reduce(tf.distribute.ReduceOp.SUM , per_replica_loss , axis = None)

    @tf.function
    def distribute_val_step(dist_data):
        per_replica_loss = strategy.run(val_step , args=(dist_data, ))
        return strategy.reduce(tf.distribute.ReduceOp.SUM , per_replica_loss , axis = None)

    def train_model_for_one_epoch (epoch , global_step):
        losses = []
        step = 1
        for batch in train_dataset :
            master_callback.on_batch_begain(global_step[0])
            loss = distributed_train_step(batch)
            losses.append(loss)
            iou_dict = {}
            dice_dict = {}
            class_names = config['data']['class_names']
            class_ious  =  per_class_iou.result()
            class_dices = per_class_dice.result()
            for i in range(len(class_ious.numpy())):
                iou_dict[class_names[i]] = class_ious[i]
                dice_dict[class_names[i]] = class_dices[i]
            data = {
                'loss' : loss ,
                'metrics' : {
                    'per_class_iou':iou_dict ,
                    'per_class_dice':dice_dict
                }
            }
            global_step[0]+=1
            step+=1
            master_callback.on_batch_end(step , epoch , data)
        return losses

    def val_model_for_one_epoch():
        val_losses = []
        for batch in val_dataset:
            val_loss  = distribute_val_step(batch)
            val_losses.append(val_loss)
        return val_losses

    stop_training = False
    start = master_callback.on_train_begain()
    global_step = [start*config['checkpoint']['batches_per_epoch']+1]
    for epoch in range(start ,  config['checkpoint']['total_epoch']):
        master_callback.on_epoch_begain(epoch)
        losses = train_model_for_one_epoch(epoch , global_step)
        val_losses = val_model_for_one_epoch()
        avg_loss = tf.reduce_mean(losses)
        avg_val_loss  = tf.reduce_mean(val_losses)
        iou_dict = {}
        dice_dict = {}
        val_iou_dict = {}
        val_dice_dict = {}
        class_names = config['data']['class_names']
        class_ious = per_class_iou.result()
        class_dices = per_class_dice.result()
        class_ious_val = per_class_iou_val.result()
        class_dices_val = per_class_dice_val.result()
        for i in range(len(per_class_iou.result())):
            iou_dict[class_names[i]] =class_ious[i]
            dice_dict[class_names[i]] = class_dices[i]
            val_iou_dict[class_names[i]]=class_ious_val[i]
            val_dice_dict[class_names[i]] = class_dices_val[i]
        data = {
            'loss' : avg_loss ,
            'val_loss' : avg_val_loss ,
            'metrics'  : {
                'per_class_iou' : iou_dict ,
                'per_class_dice': dice_dict ,
                'per_class_iou_val' : val_iou_dict ,
                'per_class_dice_val' : val_dice_dict
            },
        }
        per_class_iou.reset_state()
        per_class_dice.reset_state()
        per_class_iou_val.reset_state()
        per_class_dice_val.reset_state()
        stop_training = master_callback.on_epoch_end(epoch , data)
        if stop_training :
            break