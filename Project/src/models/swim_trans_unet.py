import sys, os
import tensorflow as tf

from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy("mixed_float16")

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

class Decoder_Block(tf.keras.layers.Layer):
    def __init__(self , filters , block_name , dropout = 0.3) :
        super().__init__()
        self.conv_layer =  convolutional_Block(filters , block_name)
        self.conv_tranpose = tf.keras.layers.Conv3DTranspose(filters = filters , kernel_size=(2,2,2) , strides=(2,2,2) , padding='same' , activation='relu')
        self.dropout=tf.keras.layers.Dropout(rate = dropout)
        self.concat = tf.keras.layers.Concatenate()

    def call(self, conv, inputs):
        u = self.conv_tranpose(inputs)

        # Crop if shapes do not match (common issue in U-Nets)
        if u.shape[1:4] != conv.shape[1:4]:
            conv_shape = tf.shape(conv)
            u = u[:, :conv_shape[1], :conv_shape[2], :conv_shape[3], :]

        c = self.concat([u, conv])
        c = self.dropout(c)
        c = self.conv_layer(c)
        return c


def pad_if_needed(x, window_shape):
    shape = tf.shape(x)
    B, D, H, W, C = shape[0], shape[1], shape[2], shape[3], shape[4]
    pad_d = (window_shape[0] - D % window_shape[0]) % window_shape[0]
    pad_h = (window_shape[1] - H % window_shape[1]) % window_shape[1]
    pad_w = (window_shape[2] - W % window_shape[2]) % window_shape[2]
    paddings = [[0, 0], [0, pad_d], [0, pad_h], [0, pad_w], [0, 0]]
    x = tf.cond(
        tf.logical_or(tf.logical_or(pad_d > 0, pad_h > 0), pad_w > 0),
        lambda: tf.pad(x, paddings),
        lambda: x
    )
    return x, (pad_d, pad_h, pad_w)

def unpad_if_needed(x, pads):
    pad_d, pad_h, pad_w = pads
    x = tf.cond(
        tf.logical_or(tf.logical_or(tf.greater(pad_d, 0), tf.greater(pad_h, 0)), tf.greater(pad_w, 0)),
        lambda: x[:, :tf.shape(x)[1]-pad_d, :tf.shape(x)[2]-pad_h, :tf.shape(x)[3]-pad_w, :],
        lambda: x
    )
    return x

def cyclic_shift(x, shift_size):
    return tf.roll(x, shift=shift_size, axis=[1, 2, 3])

def windows_partition(x, window_shape):
    b, d, h, w, c = tf.unstack(tf.shape(x))
    x = tf.reshape(x, (b, d // window_shape[0], window_shape[0],
                          h // window_shape[1], window_shape[1],
                          w // window_shape[2], window_shape[2],
                          c))
    x = tf.transpose(x, [0, 1, 3, 5, 2, 4, 6, 7])
    windows = tf.reshape(x, (-1, window_shape[0] * window_shape[1] * window_shape[2], c))
    return windows

def window_merge(windows, original_shape, window_shape):
    b, d, h, w, c = original_shape
    x = tf.reshape(windows, (b, d // window_shape[0], h // window_shape[1], w // window_shape[2],
                              window_shape[0], window_shape[1], window_shape[2], c))
    x = tf.transpose(x, [0, 1, 4, 2, 5, 3, 6, 7])
    return tf.reshape(x, (b, d, h, w, c))

class MLP(tf.keras.layers.Layer):
    def __init__(self, emb_dim):
        super().__init__()
        self.fc1 = tf.keras.layers.Dense(emb_dim * 4, activation='gelu')
        self.fc2 = tf.keras.layers.Dense(emb_dim)
    def call(self, x):
        return self.fc2(self.fc1(x))

class SwinTransformer(tf.keras.layers.Layer):
    def __init__(self, dim, window_shape, num_heads, dropout=0.1):
        super().__init__()
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.attn = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=dim // num_heads, dropout=dropout)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.mlp = MLP(emb_dim=dim)
        self.window_shape = window_shape
    def call(self, x):
        shortcut = x
        shape = tf.shape(x)
        B, D, H, W, C = shape[0], shape[1], shape[2], shape[3], shape[4]
        x, pads = pad_if_needed(x, self.window_shape)
        padded_shape = tf.shape(x)
        padded_D, padded_H, padded_W = padded_shape[1], padded_shape[2], padded_shape[3]
        x = self.norm1(x)
        x = windows_partition(x, self.window_shape)
        x = self.attn(x, x)
        x = window_merge(x, [B, padded_D, padded_H, padded_W, C], self.window_shape)
        x = unpad_if_needed(x, pads)
        x = shortcut + x
        shortcut = x
        x = self.norm2(x)
        x = self.mlp(x)
        return shortcut + x

class SwinTransformerBlock(tf.keras.layers.Layer):
    def __init__(self, dim, window_shape, num_heads, shift_size, use_checkpointing=False):
        super().__init__()
        self.block1 = SwinTransformer(dim, window_shape, num_heads)
        self.block2 = SwinTransformer(dim, window_shape, num_heads)
        self.shift_size = shift_size
        self.use_checkpointing = use_checkpointing

    def call(self, x):
        def _inner_call(x):
            x = self.block1(x)
            x = cyclic_shift(x, self.shift_size)
            x = self.block2(x)
            x = cyclic_shift(x, [-s for s in self.shift_size])
            return x

        if self.use_checkpointing:
            return tf.recompute_grad(_inner_call)(x)
        else:
            return _inner_call(x)

class SwinTransformerBottleneck(tf.keras.layers.Layer):
    def __init__(self, config):
        super().__init__()
        bottleneck_config = config['model']['bottleneck']
        dim = bottleneck_config['dim']
        use_checkpointing = config['model'].get('use_checkpointing', False)

        self.blocks = [SwinTransformerBlock(
            dim=dim,
            window_shape=tuple(bottleneck_config['window_shape']),
            num_heads=bottleneck_config['num_heads'],
            shift_size=tuple(bottleneck_config['shift_size']),
            use_checkpointing=use_checkpointing
        ) for _ in range(bottleneck_config['depth'])]

    def call(self, inputs):
        x = inputs
        for block in self.blocks:
            x = block(x)
        return x

class HybridDecoder(tf.keras.models.Model):
    def __init__(self , config) :
        super().__init__()
        num_classes = config['data']['num_classes']
        self.SwinTransformer = SwinTransformerBottleneck(config)
        self.Decoder_Block1 = Decoder_Block(256 , "decoder_block_256")
        self.Decoder_Block2 = Decoder_Block(128,'decoder_block_128')
        self.Decoder_Block3 = Decoder_Block(64 , "decoder_block_64")
        # Output raw logits for numerical stability with the loss function
        self.output_conv = tf.keras.layers.Conv3D(filters=num_classes ,kernel_size=(1,1,1), padding='same', name='output_head', dtype='float32')

    def call(self, convs ) :
        f1, f2, f3, f4 = convs
        x = self.SwinTransformer(f4)

        x = self.Decoder_Block1(f3 , x)
        x = self.Decoder_Block2(f2 , x)
        x = self.Decoder_Block3(f1 , x)
        x = self.output_conv(x)
        return x

class SwinTransUnet(tf.keras.models.Model):
    def __init__(self, config) :
        super().__init__()
        self.encoder = encoder()
        self.decoder = HybridDecoder(config)

    def call(self, inputs) :
        convs , _ = self.encoder(inputs)
        f1, f2, f3, f4 = convs
        x = self.decoder((f1, f2, f3, f4))
        return x
