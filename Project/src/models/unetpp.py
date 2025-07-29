import sys, os
import tensorflow as tf 

from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy("mixed_float16")
# # Optional: Mount Drive
# from google.colab import drive
# drive.mount('/content/drive')

# Add repo + Project folder to path
repo_path = "/content/drive/MyDrive/Medical-Segmentation-Decathlon"
sys.path.append(repo_path)
sys.path.append(os.path.join(repo_path, "Project"))




class convolutional_Block(tf.keras.layers.Layer):
    def __init__(self, n_filters, block_name, n_layers=2, activation='relu', kernel_size=(3, 3, 3)):
        super().__init__()
        self.conv_layers = []
        for i in range(n_layers):
            safe_name = block_name.replace(" ", "_").replace("-", "_")
            self.conv_layers.append(tf.keras.Sequential([
                tf.keras.layers.Conv3D(filters=n_filters, kernel_size=kernel_size, padding='same',
                                    activation=None, name=f'conv_{safe_name}_{i}'),
                tf.keras.layers.BatchNormalization(name=f'bn_{safe_name}_{i}'),
                tf.keras.layers.Activation(activation, name=f'act_{safe_name}_{i}')
            ]))


    def call(self, inputs):
        x = inputs
        for layer in self.conv_layers:
            x = layer(x)
        return x


class encoder_block(tf.keras.layers.Layer):
    def __init__(self, n_filters, block_name, pool_size=(2, 2, 2), dropout=0.3):
        super().__init__()
        self.convolutional_Block = convolutional_Block(n_filters=n_filters, block_name=block_name)
        self.pool = tf.keras.layers.MaxPooling3D(pool_size=pool_size, name=f'pool_layer_{block_name}')
        self.dropout = tf.keras.layers.Dropout(rate=dropout)

    def call(self, inputs):
        f = self.convolutional_Block(inputs)
        p = self.pool(f)
        p = self.dropout(p)
        return f, p


class encoder(tf.keras.models.Model):
    def __init__(self):
        super().__init__()
        self.encoder_block_1 = encoder_block(64, "block_1")
        self.encoder_block_2 = encoder_block(128, "block_2")
        self.encoder_block_3 = encoder_block(256, "block_3")
        self.encoder_block_4 = encoder_block(512, "block_4")

    def call(self, inputs):
        f1, p1 = self.encoder_block_1(inputs)
        f2, p2 = self.encoder_block_2(p1)
        f3, p3 = self.encoder_block_3(p2)
        f4, p4 = self.encoder_block_4(p3)
        return (f1, f2, f3, f4), (p1, p2, p3, p4)

class BottleNeck(tf.keras.models.Model): 
    def __init__(self) : 
        super().__init__()
        self.BottleNeck_layer = tf.keras.layers.Conv3D(filters=1024 , kernel_size= (2,2,2) , activation='relu' , padding = 'same' , name = "bottleneck_layer")
    def call(self, inputs ): 
        x = self.BottleNeck_layer(inputs)
        return x 
class Decoder_Block(tf.keras.layers.Layer): 
    def __init__(self , filters , block_name , dropout = 0.3) :
        super().__init__()
        self.conv_layer =  convolutional_Block(filters , block_name) 
        self.conv_tranpose = tf.keras.layers.Conv3DTranspose(filters = filters , kernel_size=(2,2,2) , strides=(2,2,2) , padding='same' , activation='relu')
        self.dropout=tf.keras.layers.Dropout(rate = dropout) 
        self.concat = tf.keras.layers.Concatenate() 
    def build(self, input_shape):
        super().build(input_shape)
    def call(self, conv, inputs):
        u = self.conv_tranpose(inputs)

 
        if u.shape[1:4] != conv.shape[1:4]:
            u = tf.image.resize(u, size=[conv.shape[1], conv.shape[2], conv.shape[3]], method='trilinear')
        
        c = self.concat([u, conv])
        c = self.dropout(c)
        c = self.conv_layer(c)
        return c
class Decoder(tf.keras.models.Model): 
    def __init__(self): 
        super().__init__() 

        self.decoder_block_01 = Decoder_Block(64 , 'Decoder_01')
        self.decoder_block_02 = Decoder_Block(64 , 'Decoder_02')
        self.decoder_block_03 = Decoder_Block(64 , 'Decoder_03')
        self.decoder_block_04 = Decoder_Block(64 , 'Decoder_04') 
        self.decoder_block_11 = Decoder_Block(128 , 'Decoder_11') 
        self.decoder_block_12 = Decoder_Block(128 , "Decoder_12") 
        self.decoder_block_13 = Decoder_Block(128 , 'Decoder_13') 
        self.decoder_block_21 = Decoder_Block(256 , "Decoder_21") 
        self.decoder_block_22 = Decoder_Block(256 , "Decoder_22") 
        self.decoder_block_31 = Decoder_Block(512 , "Decoder_31")

        self.concat_02 = tf.keras.layers.Concatenate() 
        self.concat_03 = tf.keras.layers.Concatenate() 
        self.concat_04 = tf.keras.layers.Concatenate() 
        self.concat_12 = tf.keras.layers.Concatenate() 
        self.concat_13 = tf.keras.layers.Concatenate() 
        self.concat_22 = tf.keras.layers.Concatenate() 

        self.output_o1 = tf.keras.layers.Conv3D(4, (1,1,1), padding='same',
            name="output_head_1", activation="softmax")
        self.output_o2 = tf.keras.layers.Conv3D(4, (1,1,1), padding='same',
            name="output_head_2", activation="softmax")
        self.output_o3 = tf.keras.layers.Conv3D(4, (1,1,1), padding='same',
            name="output_head_3", activation="softmax")
        self.output_o4 = tf.keras.layers.Conv3D(4, (1,1,1), padding='same',
            name="output_head_4", activation="softmax")
    


    def build(self, input_shape):
        # input_shape will be a tuple like ((B,D1,H1,W1,C1), …, (B,D5,H5,W5,C5))
        # We don’t need to manually build sublayers here—just tell Keras “I’m built”
        super().build(input_shape)

    def call (self, convs ): 
        f1,f2,f3,f4,f5=convs 
        decoder_31=self.decoder_block_31(f4,f5)
        decoder_21 = self.decoder_block_21(f3,f4)
        decoder_11 = self.decoder_block_11(f2,f3)
        deocder_01 = self.decoder_block_01(f1,f2) 

        concat_22 = self.concat_22([f3,decoder_21]) 
        decoder_22 = self.decoder_block_22(concat_22 , decoder_31)

        concat_12 = self.concat_12([f2,decoder_11])
        decoder_12 = self.decoder_block_12(concat_12,decoder_21)

        concat_02 = self.concat_02([f1,deocder_01]) 
        decoder_02 = self.decoder_block_02(concat_02 , decoder_11)

        concat_13 = self.concat_13([f2,decoder_11,decoder_12])
        decoder_13 = self.decoder_block_13(concat_13,decoder_22)


        concat_03 = self.concat_03([f1 , deocder_01, decoder_02])
        decoder_03 = self.decoder_block_03(concat_03 , decoder_12)


        concat_04 = self.concat_04([f1 , deocder_01 , decoder_02,decoder_03])
        decoder_04 = self.decoder_block_04(concat_04 , decoder_13) 

        output_01 = self.output_o1(deocder_01)
        output_02 = self.output_o2(decoder_02)
        output_03 = self.output_o3(decoder_03)
        output_04 = self.output_o4(decoder_04)

        return output_01 , output_02, output_03, output_04 


class UNET_PLUS_PLUS(tf.keras.models.Model): 
    def __init__(self): 
        super().__init__()
        self.encoder  = encoder()
        self.bottleneck = BottleNeck() 
        self.decoder = Decoder() 

    def  call(self, inputs ): 
        convs , pools =self.encoder(inputs)
        p1,p2,p3,p4=pools
        f5=self.bottleneck(p4)
        f1,f2,f3,f4=convs
        output_01 , output_02,output_03,output_04=self.decoder((f1,f2,f3,f4,f5))  
        return output_01 , output_02, output_03, output_04



