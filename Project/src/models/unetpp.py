import sys, os
import tensorflow as tf 

from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy("mixed_bfloat16")

repo_path = "/content/drive/MyDrive/Medical-Segmentation-Decathlon"
sys.path.append(repo_path)





class convolutional_Block(tf.keras.layers.Layer):
    def __init__(self, n_filters, block_name, n_layers=2, activation='relu', kernel_size=(3, 3, 3)):
        super().__init__()
        self.conv_layers = []
        for i in range(n_layers):
            safe_name = block_name.replace(" ", "_").replace("-", "_")
            self.conv_layers.append(tf.keras.Sequential([
                tf.keras.layers.Conv3D(filters=n_filters, kernel_size=kernel_size, padding='same',
                                    activation=None, name=f'conv_{safe_name}_{i}'),
                tf.keras.layers.LayerNormalization(name=f'bn_{safe_name}_{i}'),
                tf.keras.layers.Activation(activation, name=f'act_{safe_name}_{i}')
            ]))


    def call(self, inputs,training = True ):
        x = inputs
        for layer in self.conv_layers:
            x = layer(x , training = training )
        return x


class encoder_block(tf.keras.layers.Layer):
    def __init__(self, n_filters, block_name, pool_size=(2, 2, 2), dropout=0.3):
        super().__init__()
        self.convolutional_Block = convolutional_Block(n_filters=n_filters, block_name=block_name)
        self.pool = tf.keras.layers.MaxPooling3D(pool_size=pool_size, name=f'pool_layer_{block_name}')
        self.dropout = tf.keras.layers.Dropout(rate=dropout)

    def call(self, inputs, training=True ):
        f = self.convolutional_Block(inputs ,training = training)
        p = self.pool(f)
        p = self.dropout(p,training = training)
        return f, p


class encoder(tf.keras.models.Model):
    def __init__(self):
        super().__init__()
        self.encoder_block_1 = encoder_block(64, "block_1")
        self.encoder_block_2 = encoder_block(128, "block_2")
        self.encoder_block_3 = encoder_block(256, "block_3")
        self.encoder_block_4 = encoder_block(512, "block_4")

    def call(self, inputs , training = True  ):
        f1, p1 = self.encoder_block_1(inputs , training =training )
        f2, p2 = self.encoder_block_2(p1,training = training )
        f3, p3 = self.encoder_block_3(p2,training = training )
        f4, p4 = self.encoder_block_4(p3 ,training = training )
        return (f1, f2, f3, f4), (p1, p2, p3, p4)

class BottleNeck(tf.keras.models.Model): 
    def __init__(self) : 
        super().__init__()
        self.BottleNeck_layer = tf.keras.layers.Conv3D(filters=1024 , kernel_size= (2,2,2) , activation='relu' , padding = 'same' , name = "bottleneck_layer")
    def call(self, inputs , training=True ): 
        x = self.BottleNeck_layer(inputs )
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
    def call(self, conv, inputs , training = True ):
        u = self.conv_tranpose(inputs)

 
        if u.shape[1:4] != conv.shape[1:4]:

            u_shape = tf.shape(u)
            conv_shape = tf.shape(conv)

            target_h, target_w, target_d = conv_shape[1], conv_shape[2], conv_shape[3]

            u_reshaped = tf.transpose(u, perm=[0, 3, 1, 2, 4])
            u_reshaped = tf.reshape(u_reshaped, (u_shape[0] * u_shape[3], u_shape[1], u_shape[2], u_shape[4]))


            u_resized_2d = tf.image.resize(u_reshaped, size=(target_h, target_w), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

            u_resized_5d = tf.reshape(u_resized_2d, (u_shape[0], u_shape[3], target_h, target_w, u_shape[4]))
            u = tf.transpose(u_resized_5d, perm=[0, 2, 3, 1, 4])

            u_reshaped_for_depth = tf.transpose(u, perm=[0, 1, 3, 2, 4])
            u_reshaped_for_depth = tf.reshape(u_reshaped_for_depth, (u_shape[0] * target_h, u_shape[3], target_w, u_shape[4]))

            u_resized_depth = tf.image.resize(u_reshaped_for_depth, size=(target_d, target_w), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

            u_final_shape = tf.reshape(u_resized_depth, (u_shape[0], target_h, target_d, target_w, u_shape[4]))
            u = tf.transpose(u_final_shape, perm=[0, 1, 3, 2, 4])
        
        c = self.concat([u, conv])

        c = self.dropout(c,training = training)
        c = self.conv_layer(c ,  training=training)
        return c
class Decoder(tf.keras.models.Model): 
    def __init__(self , num_classes): 
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

        self.output_o1 = tf.keras.layers.Conv3D(num_classes, (1,1,1), padding='same',
            name="output_head_1",dtype ='float32')
        self.output_o2 = tf.keras.layers.Conv3D(num_classes, (1,1,1), padding='same',
            name="output_head_2",dtype ='float32')
        self.output_o3 = tf.keras.layers.Conv3D(num_classes, (1,1,1), padding='same',
            name="output_head_3" , dtype = 'float32')
        self.output_o4 = tf.keras.layers.Conv3D(num_classes, (1,1,1), padding='same',
            name="output_head_4",dtype ='float32')
    


    def build(self, input_shape):

        super().build(input_shape)

    def call (self, convs, training = True ): 
        f1,f2,f3,f4,f5=convs 
        decoder_31=self.decoder_block_31(f4,f5 ,training =  training)
        decoder_21 = self.decoder_block_21(f3,f4 ,training = training)
        decoder_11 = self.decoder_block_11(f2,f3 ,training = training)
        deocder_01 = self.decoder_block_01(f1,f2 ,training = training) 

        concat_22 = self.concat_22([f3,decoder_21] ) 
        decoder_22 = self.decoder_block_22(concat_22 , decoder_31 ,training = training)

        concat_12 = self.concat_12([f2,decoder_11])
        decoder_12 = self.decoder_block_12(concat_12,decoder_21 ,training = training)

        concat_02 = self.concat_02([f1,deocder_01] ) 
        decoder_02 = self.decoder_block_02(concat_02 , decoder_11 ,training = training)

        concat_13 = self.concat_13([f2,decoder_11,decoder_12] )
        decoder_13 = self.decoder_block_13(concat_13,decoder_22 ,training = training)


        concat_03 = self.concat_03([f1 , deocder_01, decoder_02] )
        decoder_03 = self.decoder_block_03(concat_03 , decoder_12 ,training = training)


        concat_04 = self.concat_04([f1 , deocder_01 , decoder_02,decoder_03])
        decoder_04 = self.decoder_block_04(concat_04 , decoder_13 ,training= training) 

        output_01 = self.output_o1(deocder_01)
        output_02 = self.output_o2(decoder_02)
        output_03 = self.output_o3(decoder_03)
        output_04 = self.output_o4(decoder_04)

        return output_01 , output_02, output_03, output_04 


class UNET_PLUS_PLUS(tf.keras.models.Model): 
    def __init__(self , num_classes): 
        super().__init__()
        self.encoder  = encoder()
        self.bottleneck = BottleNeck() 
        self.decoder = Decoder( num_classes) 

    def  call(self, inputs , training = True ): 
        convs , pools =self.encoder(inputs ,training =  training  )
        p1,p2,p3,p4=pools
        f5=self.bottleneck(p4,training =  training)
        f1,f2,f3,f4=convs
        output_01 , output_02,output_03,output_04=self.decoder((f1,f2,f3,f4,f5) ,training =  training)  
        return output_01 , output_02, output_03, output_04