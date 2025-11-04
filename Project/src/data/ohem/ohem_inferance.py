import tensorflow as tf
from tqdm import tqdm
import pandas as pd
import sys

repo_path = "/content/drive/MyDrive/Medical-Segmentation-Decathlon"
sys.path.append(repo_path)

class OHEM_Inference:
    """
    Orchestrates the Online Hard Example Mining (OHEM) inference phase.

    This class consumes a "mining" dataset, runs a model to calculate the
    per-patch loss, and identifies the top K% of patches with the highest loss.
    It is designed to be fully compatible with tf.distribute.TPUStrategy.
    """
    def __init__(self, config, tpu_dataset ,metadata_dataset , model, loss_fn, strategy):
        """
        Initializes the OHEM inference engine.

        Args:
            config (dict): The main configuration dictionary.
            mining_dataset (tf.data.Dataset): The dataset from OHEMDataPipeline.
            model (tf.keras.Model): The trained model to use for inference.
            loss_fn: The loss function to score patch difficulty.
            strategy: The active tf.distribute.Strategy.
        """
        self.config = config
        self.tpu_dataset = tpu_dataset
        self.metadata_dataset = metadata_dataset
        self.model = model
        self.loss_fn = loss_fn
        self.strategy = strategy
    
    @tf.function 
    def _inference_step(self, dist_input):
        """
        Performs a forward pass and calculates per-patch loss on a single replica.

        This function is executed on each TPU core. It computes the loss for its
        local batch and returns the loss along with the metadata.

        Args:
            dist_input (tuple): A tuple of tensors for the local batch, containing
                                (image_patch, label_patch, volume_id, patch_coord).

        Returns:
            A tuple of tensors: (per_example_loss, volume_id, patch_coord).
        """
        image_patch, label_patch= dist_input


        if self.config['model']['name'] == 'unet_plus_plus':
            model_output = self.model(image_patch, training=False)
            per_example_loss = self.loss_fn(label_patch, model_output)
        else:
            model_logits = self.model(image_patch, training=False)
            per_example_loss = self.loss_fn(label_patch, model_logits)


        return per_example_loss

    def __call__(self, top_k_percent=0.20):
        """
        Executes the full OHEM mining and ranking process.

        This is the main entry point that orchestrates the distributed inference,
        gathers the results, and returns the final lightweight metadata list.

        Args:
            top_k_percent (float): The percentage of hardest patches to mine.

        Returns:
            list: A list of metadata tuples [(volume_id, coordinates), ...],
                  where volume_id is a byte string and coordinates is a NumPy array.
        """
        print("⛏️  Starting OHEM mining phase...")
        all_patch_info = []

 
        dist_dataset = self.strategy.experimental_distribute_dataset(self.tpu_dataset)

     
        for batch , metadata_batch in tqdm(zip(dist_dataset , self.metadata_dataset), desc="Mining Hard Patches"):
            per_replica_losses= self.strategy.run(
                self._inference_step, args=(batch,)
            )


            gathered_losses = self.strategy.gather(per_replica_losses, axis=0)
            gathered_paths = metadata_batch[0]
            gathered_coords = metadata_batch[1]

          
            for i in range(gathered_losses.shape[0]):
                all_patch_info.append({
                    "loss": gathered_losses[i].numpy(),
                    "volume_id": gathered_paths[i].numpy(), 
                    "coords": gathered_coords[i].numpy()
                })

        print(f"✅ Mined a total of {len(all_patch_info)} patches.")
        

        all_patch_info.sort(key=lambda x: x["loss"], reverse=True)


        num_to_keep = int(len(all_patch_info) * top_k_percent)
        hardest_patches = all_patch_info[:num_to_keep]


        hard_patch_metadata = [
            (p["volume_id"], p["coords"]) for p in hardest_patches
        ]

        print(f"✅ OHEM mining complete. Identified {len(hard_patch_metadata)} hard patches.")
        return hard_patch_metadata