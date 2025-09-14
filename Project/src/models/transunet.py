import sys, os
import tensorflow as tf

from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy("mixed_float16")



repo_path = "/content/drive/MyDrive/Medical-Segmentation-Decathlon"
sys.path.append(repo_path)


from Project.src.models.unetpp import *

class multi_layer_preceptron_part(tf.keras.layers.Layer):
    def __init__(self,d_model):
        super().__init__()
        self.layer1 = tf.keras.layers.Dense(d_model*4 , activation='gelu' ,  kernel_regularizer=tf.keras.regularizers.l2(1e-5))
        self.layer2 = tf.keras.layers.Dense(d_model ,  kernel_regularizer=tf.keras.regularizers.l2(1e-5))
    def call(self, inputs ):
        x=self.layer1(inputs)
        x = self.layer2(x)
        return x


class Transformer_encoder(tf.keras.layers.Layer):
    def __init__(self , num_heads = 8 , key_dim=64 , d_model=512  , dropout_rate =0.1):
        super().__init__()
        self.multihead_attention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads , key_dim=key_dim)
        self.attn_dropout = tf.keras.layers.Dropout(dropout_rate)
        self.ffn_dropout = tf.keras.layers.Dropout(dropout_rate)
        self.add1 = tf.keras.layers.Add()
        self.add2 = tf.keras.layers.Add()
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.mhp = multi_layer_preceptron_part(d_model=d_model)
    def call(self,inputs):
        x =inputs
        x=self.multihead_attention(x,x,x)
        x=self.attn_dropout(x)
        x=self.add1([inputs,x])
        p=self.norm1(x)
        x=self.mhp(p)
        x= self.ffn_dropout(x)
        x=self.add2([p,x])
        x=self.norm2(x)
        return x


class Transformer_part(tf.keras.layers.Layer):
    def __init__(self, d_model=512, num_layers=12):
        super().__init__()
        self.projection = tf.keras.layers.Dense(d_model)
        self.d_model = d_model
        self.transformer_blocks = [Transformer_encoder() for _ in range(num_layers)]
        self.positional_embedding = None

    def build(self, input_shape):

        H, W, D = input_shape[1], input_shape[2], input_shape[3]
        num_patches = H * W * D  
        self.positional_embedding = tf.keras.layers.Embedding(
            input_dim=num_patches, output_dim=self.d_model
        )
        super().build(input_shape)

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        H, W, D, C = inputs.shape[1], inputs.shape[2], inputs.shape[3], inputs.shape[4]
        num_patches = H * W * D


        x = tf.reshape(inputs, (batch_size, num_patches, C))


        linear_projection = self.projection(x)

        position = tf.range(start=0, limit=num_patches, delta=1)
        pos_emb = self.positional_embedding(position)  
        pos_emb = tf.expand_dims(pos_emb, 0)
        x = linear_projection + pos_emb


        for block in self.transformer_blocks:
            x = block(x)


        x = tf.reshape(x, (batch_size, H, W, D, self.d_model))
        return x

class Decoder(tf.keras.models.Model):
    def __init__(self , num_classes) :
        super().__init__( )
        self.Transformer = Transformer_part()
        self.Decoder_Block1 = Decoder_Block(256 , "decoder_block_256")
        self.Decoder_Block2 =Decoder_Block(128,'decoder_block_128')
        self.Decoder_Block3 = Decoder_Block(64 , "decoder_block_64")
        self.output_conv = tf.keras.layers.Conv3D(filters=num_classes ,kernel_size=(1,1,1), padding='same' , name='output_head')

    def call(self, convs ) :
        f1,f2,f3,f4=convs
        x = self.Transformer(f4)
        x = self.Decoder_Block1(f3 , x)
        x = self.Decoder_Block2(f2 , x)
        x = self.Decoder_Block3(f1 , x)
        x = self.output_conv(x)
        return x
class TRANSUNET(tf.keras.models.Model):
    def __init__(self , num_classes) :
        super().__init__()
        self.encoder =encoder()
        self.decoder = Decoder(num_classes)
    def call(self, inputs) :
        convs , pools=self.encoder(inputs)
        x=self.decoder(convs)
        return x



