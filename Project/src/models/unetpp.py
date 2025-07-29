import sys, os

# Optional: Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Add repo + Project folder to path
repo_path = "/content/drive/MyDrive/Medical-Segmentation-Decathlon"
sys.path.append(repo_path)
sys.path.append(os.path.join(repo_path, "Project"))

import tensorflow as tf 

class convolutional_Block_For_Unet_Plus_Plus(tf.keras.models.Model):
  def __init__(self, n_filters,n_layers , block_name , activation = 'relu',kernel_size=(3,3)):
    super().__init__() 
    self.conv_layers = [tf.keras.layers.Conv2D(filters=n_filters , kernel_size=kernel_size , activation= activation) for _ in range(n_layers)]
  def call(self , inputs): 
    x = inputs 
    for layer in self.conv_layers: 
      x=layer(x)
    return x

class encoder_for_unet_plus_plus(tf.keras.models.Model): 
  def __init__(self , input_shape ):
    pass 