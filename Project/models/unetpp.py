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

class encoder(tf.keras.models.Model): 
  def __init__(self , input_shape ,dropout_rate=0.3 ):
    base = tf.keras.applications.resnet50.ResNet50(include_top=False , weights="imagenet" , input_shape=input_shape) 
    self.layer_names = [
            'conv1_relu',
            'conv2_block3_out',
            'conv3_block4_out',
            'conv4_block6_out'
        ]
    self.backbone = tf.keras.models.Model(inputs = base.input , outputs = [base.get_layer(layer) for layer in self.layer_names])
    self.pool = [tf.keras.layers.MaxPooling2D((2,2) , name = f"encdoer pool - {i+1}") for i in range(4)]

  def call(self , inputs ) : 
    feats = self.backbone(inputs)
    pools = [] 
    for i , feat in enumerate(feats): 
      p=self.pool[i](feat)
      p = self.backbone[i](p)
      pools.append(p)

    return feats , pools
