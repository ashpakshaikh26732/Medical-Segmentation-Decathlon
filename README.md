# 3D Medical Image Segmentation with Advanced Deep Learning Architectures

**Author:** Ashpak Jabbar Shaikh


## 1. Project Overview

This repository contains a professional-grade, end-to-end pipeline for 3D medical image segmentation, developed in TensorFlow and Keras. The project is designed as a robust and scalable framework for tackling the ten diverse challenges of the **Medical Segmentation Decathlon (MSD)**.

The entire system is built from the ground up, featuring a custom, distributed training loop optimized for TPUs, a highly configurable data pipeline, multiple state-of-the-art model architectures, and a sophisticated callback system for comprehensive experiment management and monitoring.

The core philosophy of this project is **modularity and reusability**. Every component, from data loading to model training, is designed to be configurable through YAML files, allowing for easy adaptation to each of the ten MSD tasks and beyond without significant code changes.


## 2. The Medical Segmentation Decathlon Challenge

This framework is designed to be a general-purpose solution, adaptable to all ten tasks from the Medical Segmentation Decathlon. The `DataPipeline` is specifically engineered to handle both MRI and CT modalities with appropriate, task-specific preprocessing, making it a versatile tool for the entire challenge.

The ten distinct segmentation challenges are as follows:

|   |
| - |
|   |

|             |                 |                    |              |                                    |
| ----------- | --------------- | ------------------ | ------------ | ---------------------------------- |
| **Task ID** | **Task Name**   | **Target Anatomy** | **Modality** | **Foreground Classes**             |
| **Task01**  | Brain Tumour    | Brain              | MRI          | 2 (Tumor Core, Edema)              |
| **Task02**  | Heart           | Heart              | MRI          | 1 (Left Atrium)                    |
| **Task03**  | Liver           | Liver              | CT           | 2 (Liver, Tumors)                  |
| **Task04**  | Hippocampus     | Hippocampus        | MRI          | 2 (Anterior, Posterior)            |
| **Task05**  | Prostate        | Prostate           | MRI          | 2 (Peripheral Zone, Central Gland) |
| **Task06**  | Lung            | Lung               | CT           | 1 (Lung Nodules)                   |
| **Task07**  | Pancreas        | Pancreas           | CT           | 2 (Pancreas, Tumors)               |
| **Task08**  | Hepatic Vessels | Liver              | CT           | 2 (Vessels, Tumors)                |
| **Task09**  | Spleen          | Spleen             | CT           | 1 (Spleen)                         |
| **Task10**  | Colon           | Colon              | CT           | 1 (Colon Cancer)                   |


### Task 01: Brain Tumour

- **Target Anatomy:** Brain

- **Modality:** Multi-modal Magnetic Resonance Imaging (MRI) - (T1, T1Gd, T2, FLAIR)

- **Foreground Classes (2):**

  1. **Tumor Core:** A merged class representing the necrotic/non-enhancing core and the enhancing tumor.

  2. **Peritumoral Edema:** The swelling surrounding the tumor.

- **Challenge:** Handling multi-modal 4D input and severe class imbalance.


### Task 02: Heart

- **Target Anatomy:** Heart

- **Modality:** Magnetic Resonance Imaging (MRI)

- **Foreground Classes (1):**

  1. **Left Atrium:** Segmenting the left atrial chamber of the heart.

- **Challenge:** High anatomical variability between patients.


### Task 03: Liver

- **Target Anatomy:** Liver

- **Modality:** Computed Tomography (CT)

- **Foreground Classes (2):**

  1. **Liver:** Segmenting the entire liver organ.

  2. **Tumors:** Segmenting cancerous lesions within the liver.

- **Challenge:** Low contrast between the liver and its tumors, requiring careful normalization (HU clipping).


### Task 04: Hippocampus

- **Target Anatomy:** Hippocampus (Brain)

- **Modality:** Magnetic Resonance Imaging (MRI)

- **Foreground Classes (2):**

  1. **Anterior Hippocampus:** Segmenting the head of the hippocampus.

  2. **Posterior Hippocampus:** Segmenting the body and tail of the hippocampus.

- **Challenge:** Segmenting very small, complex, and elongated structures.


### Task 05: Prostate

- **Target Anatomy:** Prostate

- **Modality:** Magnetic Resonance Imaging (MRI)

- **Foreground Classes (2):**

  1. **Peripheral Zone:** The outer region of the prostate.

  2. **Central Gland:** The inner region of the prostate.

- **Challenge:** Differentiating between zones with subtle texture differences.


### Task 06: Lung

- **Target Anatomy:** Lung

- **Modality:** Computed Tomography (CT)

- **Foreground Classes (1):**

  1. **Lung Nodules:** Segmenting small, often spherical cancerous nodules.

- **Challenge:** Detecting and segmenting very small target objects within a large volume.


### Task 07: Pancreas

- **Target Anatomy:** Pancreas

- **Modality:** Computed Tomography (CT)

- **Foreground Classes (2):**

  1. **Pancreas:** Segmenting the entire pancreas organ.

  2. **Tumors:** Segmenting cancerous lesions within the pancreas.

- **Challenge:** The pancreas has a highly variable shape and is often difficult to distinguish from surrounding organs.


### Task 08: Hepatic Vessels

- **Target Anatomy:** Liver

- **Modality:** Computed Tomography (CT)

- **Foreground Classes (2):**

  1. **Vessels:** Segmenting the intricate network of hepatic (liver) blood vessels.

  2. **Tumors:** Segmenting tumors near the vessels.

- **Challenge:** Segmenting fine, branching, tubular structures.


### Task 09: Spleen

- **Target Anatomy:** Spleen

- **Modality:** Computed Tomography (CT)

- **Foreground Classes (1):**

  1. **Spleen:** Segmenting the entire spleen organ.

- **Challenge:** A straightforward organ segmentation task, good for baseline testing.


### Task 10: Colon

- **Target Anatomy:** Colon

- **Modality:** Computed Tomography (CT)

- **Foreground Classes (1):**

  1. **Colon Cancer:** Segmenting cancerous masses in the colon.

- **Challenge:** High variability in the location, size, and shape of tumors.


## 3. Core Features & Technical Architecture

This project was engineered with a focus on modern deep learning best practices.

- **Modular & Scalable Structure:** The source code is logically organized by function (`models`, `losses`, `data`, `callbacks`, etc.) within a `src/` directory. This clean separation of concerns makes the project easy to navigate, maintain, and extend.

- **Multiple SOTA Architectures:** The framework includes complete implementations for several advanced 3D segmentation models, allowing for easy comparison and experimentation:

  - **UNET++:** A nested U-Net architecture that uses deep supervision to capture features at varying scales.

  - **TransUNET:** A hybrid Vision Transformer and U-Net architecture that leverages the global context-capturing power of transformers.

  - **Swin UNET:** A U-Net-like architecture built entirely from Swin Transformer blocks, offering hierarchical feature representation.

- **Configurable Data Pipeline (`DataPipeline` Class):**

  - **Dynamic & Reusable:** A single, powerful `DataPipeline` class handles the entire data loading and preprocessing workflow. It is driven by a YAML configuration file, allowing it to adapt to different datasets, modalities, and hyperparameters.

  - **Multi-Modality Support:** The pipeline is designed to handle both **MRI** and **CT** data, with modality-specific normalization logic (per-image min-max for MRI, HU clipping for CT).

  - **Dynamic Patching:** Implements a fully "soft-coded" patching mechanism that can handle any input volume size and patch size, automatically padding the volume to ensure perfect divisibility.

- **Advanced 3D Data Augmentation:** A robust set of on-the-fly 3D augmentations are applied to the training data to improve model generalization:

  - **Geometric (on full volumes):** Random Rotation, Scaling (Zoom), and Elastic Deformations.

  - **Intensity (on individual patches):** Random Brightness, Contrast, Gaussian Noise, and Gamma Correction.

- **Custom Loss Functions & Metrics:**

  - **Hybrid Loss:** A combined loss function (`Sementic_segmentation_loss`) that leverages the stability of class-weighted Sparse Categorical Cross-Entropy and the class-imbalance resilience of Dice Loss.

  - **Deep Supervision Loss:** A specialized loss (`DeepSupervisionLoss3D`) for the UNET++ model that calculates a combined loss across all four of its output heads.

  - **Custom Keras Metrics:** Stateful, per-class metrics (`PerClassIoU`, `PerClassDice`) for detailed performance monitoring during training.

- **Custom Training Loop & Callbacks:**

  - **Full Control:** A custom training loop written from scratch provides maximum control over the training and validation process, optimized for distributed training.

  - **Master Callback System:** A single `master_callback` composes all other callbacks, creating a clean and organized training script. This system manages:

    - **Checkpointing:** Robustly saves and restores the complete training state (model, optimizer, epoch) using `tf.train.CheckpointManager`.

    - **Learning Rate Scheduling:** Implements a Cosine Decay with Warmup schedule, updated at every step.

    - **Early Stopping:** Monitors validation loss to prevent overfitting.

    - **Rich Interactive Logging:** Provides a `tqdm` progress bar, live-updating `pandas` tables, and end-of-epoch `matplotlib` graphs directly in the notebook.

    - **TensorBoard Logging:** Automatically logs all losses and metrics for visualization in TensorBoard.

- **Distributed Training:** Full support for `tf.distribute.TPUStrategy` ensures high-performance, distributed training on TPUs.


## 4. Project Structure Explained

The project is organized into a clean and scalable directory structure.

    Project/
    ├── data/
    │   └── Task01_BrainTumour/       # <-- Raw dataset files (.nii.gz)
    ├── notebooks/
    │   └── experimentation.ipynb     # <-- Your Colab notebook for development and testing
    ├── src/
    │   ├── configs/                  # <-- All YAML experiment configuration files
    │   │   └── unetpp_task01.yaml
    │   ├── data/
    │   │   └── pipeline.py           # <-- Contains the DataPipeline class
    │   ├── models/
    │   │   ├── unetpp.py, transunet.py, ...
    │   ├── losses/
    │   │   └── segmentation_losses.py # <-- Contains all custom loss classes
    │   ├── metrics/
    │   │   └── segmentation_metrics.py # <-- Contains PerClassIoU and PerClassDice
    │   ├── callbacks/
    │   │   └── master_callback.py    # <-- Contains all custom callback classes
    │   └── utils/
    │       └── augmentations.py      # <-- Standalone augmentation functions
    ├── train.py                      # <-- Main script to start a training run
    ├── evaluate.py                   # <-- Main script to run post-training evaluation
    └── requirements.txt


## 5. Usage

This project is designed to be run from the command line, driven by configuration files to ensure reproducibility.


### Step 1: Configure Your Experiment

All experiment parameters are defined in a YAML file located in `src/configs/`. This file controls everything from the data source and model choice to hyperparameters and callback settings.

**Example `unetpp_task01.yaml`:**

    model:
        name: 'unet_plus_plus'

    data:
        modality: 'MRI'
        val_count: 24
        num_classes: 3
        batch: 1
        num_replicas: 8
        # ... other data parameters ...

    checkpoint:
        checkpoint_dir: 'gs://your-bucket/checkpoints/'
        total_step: 10000
        # ... other callback parameters ...

    optimizer:
        starting_lr: 1e-4
        weight_decay: 1e-5

    loss: 'deep_supervision_loss'


### Step 2: Run Training

The `train.py` script is the main entry point. It parses the specified config file, sets up the TPU strategy, instantiates all components, and begins the custom training loop.

**To run in a Colab notebook or terminal:**

    !python /path/to/Project/train.py --config /path/to/Project/src/configs/unetpp_task01.yaml


### Step 3: Run Evaluation

After training is complete, the `evaluate.py` script is used to calculate the computationally expensive metrics on the validation or test set using the best saved checkpoint.

This script will loop through each validation image one by one, perform inference (ideally with a Sliding Window strategy), and calculate metrics like **Hausdorff Distance (HD95)** and **Average Symmetric Surface Distance (ASSD)**.


## 6. Results

_(This section will be updated with final metrics, tables, and visualizations once training is complete.)_


#### Quantitative Results

|            |            |                |         |          |          |
| ---------- | ---------- | -------------- | ------- | -------- | -------- |
| **Model**  | **Class**  | **Dice Score** | **IoU** | **HD95** | **ASSD** |
| **UNET++** | Tumor Core | TBD            | TBD     | TBD      | TBD      |
|            | Edema      | TBD            | TBD     | TBD      | TBD      |


#### Qualitative Results (Sample Predictions)

_(Placeholder for images showing Ground Truth vs. Model Prediction on validation slices)_


## 7. License

This project is licensed under the MIT License - see the `LICENSE` file for details.
