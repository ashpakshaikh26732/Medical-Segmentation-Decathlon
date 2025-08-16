import sys, os
import tensorflow as tf

from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy("mixed_float16")


# Add repo + Project folder to path
repo_path = "/content/drive/MyDrive/Medical-Segmentation-Decathlon"
sys.path.append(repo_path)




class PatchEmbedding(tf.keras.layers.Layer):
    def __init__(self, patch_size=(4, 4, 4), emb_dim=96):
        super().__init__()
        self.proj = tf.keras.layers.Conv3D(filters=emb_dim, kernel_size=patch_size, strides=patch_size, padding='valid')

    def call(self, inputs):
        return self.proj(inputs)


class PatchMerging(tf.keras.layers.Layer):
    def __init__(self, output_dim):
        super().__init__()
        self.reduction = tf.keras.layers.Dense(output_dim)

    def call(self, x):

        B, D, H, W, C = tf.shape(x)
        x = tf.reshape(x, (B, D // 2, 2, H // 2, 2, W // 2, 2, C))
        x = tf.transpose(x, perm=[0, 1, 3, 5, 2, 4, 6, 7])
        x = tf.reshape(x, (B, D // 2, H // 2, W // 2, C * 8))
        return self.reduction(x)

class PatchExpanding(tf.keras.layers.Layer):
    def __init__(self, input_dim, expand_dim, expand_ratio=(2, 2, 2)):
        super().__init__()
        self.proj = tf.keras.layers.Dense(units=expand_dim * expand_ratio[0] * expand_ratio[1] * expand_ratio[2])

        self.expand_ratio = expand_ratio
        self.expand_dim = expand_dim

    def call(self, x):
        shape = tf.shape(x)
        B, D, H, W, C = shape[0], shape[1], shape[2], shape[3], shape[4]

        x = self.proj(x)


        x = tf.reshape(x, (B, D, H, W,
                           self.expand_ratio[0], self.expand_ratio[1], self.expand_ratio[2],
                           self.expand_dim))

        x = tf.transpose(x, perm=[0, 1, 4, 2, 5, 3, 6, 7])

        x = tf.reshape(x, (B,
                           D * self.expand_ratio[0],
                           H * self.expand_ratio[1],
                           W * self.expand_ratio[2],
                           self.expand_dim))
        return x



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


def cyclic_shift(x, shift_size):
    return tf.roll(x, shift=shift_size, axis=[1, 2, 3])


class MLP(tf.keras.layers.Layer):
    def __init__(self, emb_dim):
        super().__init__()
        self.fc1 = tf.keras.layers.Dense(emb_dim * 4, activation='gelu')
        self.fc2 = tf.keras.layers.Dense(emb_dim)

    def call(self, x):
        x = self.fc1(x)
        return self.fc2(x)


class SwinTransformer(tf.keras.layers.Layer):
    def __init__(self, dim, window_shape=(4, 4, 4), num_heads=16,dropout =0.1):
        super().__init__()
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.attn = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=dim // num_heads,dropout = dropout)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.mlp = MLP(emb_dim=dim)
        self.window_shape = window_shape

    def call(self, x):
        shortcut = x
        shape = tf.shape(x)
        B, D, H, W, C = shape[0], shape[1], shape[2], shape[3], shape[4]


        x = self.norm1(x)
        x = windows_partition(x, self.window_shape)

        x = self.attn(x, x)

        x = window_merge(x, [B, D, H, W, C], self.window_shape)

        x = shortcut + x

        shortcut = x
        x = self.norm2(x)
        x = self.mlp(x)
        return shortcut + x


class SwinTransformerBlock(tf.keras.layers.Layer):
    def __init__(self, dim, window_shape=(4, 4, 4), num_heads=16, shift_size=(2, 2, 2)):
        super().__init__()
        self.block1 = SwinTransformer(dim, window_shape, num_heads)
        self.block2 = SwinTransformer(dim, window_shape, num_heads)
        self.shift_size = shift_size

    def call(self, x):
        x = self.block1(x)
        x = cyclic_shift(x, self.shift_size)
        x = self.block2(x)
        x = cyclic_shift(x, [-s for s in self.shift_size])
        return x


class Encoder(tf.keras.Model):
    def __init__(self, emb_dims=[96, 192, 384, 768], window_shape=(4, 4, 4), num_heads=16):
        super().__init__()
        self.patch_embed = PatchEmbedding(patch_size=(4, 4, 4), emb_dim=emb_dims[0])

        self.stage1 = SwinTransformerBlock(emb_dims[0], window_shape, num_heads)
        self.merge1 = PatchMerging(output_dim=emb_dims[1])

        self.stage2 = SwinTransformerBlock(emb_dims[1], window_shape, num_heads)
        self.merge2 = PatchMerging(output_dim=emb_dims[2])

        self.stage3 = SwinTransformerBlock(emb_dims[2], window_shape, num_heads)
        self.merge3 = PatchMerging(output_dim=emb_dims[3])

        self.stage4 = SwinTransformerBlock(emb_dims[3], window_shape, num_heads)

    def call(self, inputs):
        x = self.patch_embed(inputs)

        f1 = self.stage1(x)
        x = self.merge1(f1)

        f2 = self.stage2(x)
        x = self.merge2(f2)

        f3 = self.stage3(x)
        x = self.merge3(f3)

        f4 = self.stage4(x)

        return (f1, f2, f3, f4)


class Decoder(tf.keras.models.Model):
    def __init__(self, emb_dims=[96, 192, 384, 768], window_shape=(4, 4, 4), num_heads=16, num_classes):
        super().__init__()

        self.patch_expand1 = PatchExpanding(input_dim=emb_dims[3], expand_dim=emb_dims[2])
        self.Decoder_stage3 = SwinTransformerBlock(dim=emb_dims[2], window_shape=window_shape, num_heads=num_heads)

        self.patch_expand2 = PatchExpanding(input_dim=emb_dims[2], expand_dim=emb_dims[1])
        self.Decoder_stage2 = SwinTransformerBlock(dim=emb_dims[1], window_shape=window_shape, num_heads=num_heads)

        self.patch_expand3 = PatchExpanding(input_dim=emb_dims[1], expand_dim=emb_dims[0])
        self.Decoder_stage1 = SwinTransformerBlock(dim=emb_dims[0], window_shape=window_shape, num_heads=num_heads)

        self.output_head = tf.keras.layers.Conv3D(filters=num_classes, kernel_size=(1, 1, 1), activation='softmax', padding='same', dtype='float32')
        self.final_upsample = PatchExpanding(input_dim=emb_dims[0], expand_dim=emb_dims[0])
        self.final_expand2 = PatchExpanding(input_dim=emb_dims[0], expand_dim=emb_dims[0])


        self.add1  = tf.keras.layers.Add()
        self.add2  = tf.keras.layers.Add()
        self.add3  = tf.keras.layers.Add()
    def call(self, skip):
        f1, f2, f3, f4 = skip

        x = self.patch_expand1(f4)
        x  = self.add1([x,f3])
        x = self.Decoder_stage3(x)

        x = self.patch_expand2(x)

        x = self.add2([x,f2])
        x = self.Decoder_stage2(x)

        x = self.patch_expand3(x)
        x = self.add1([x , f1])
        x = self.Decoder_stage1(x)
        x= self.final_upsample(x)
        x =self.final_expand2(x)
        x = self.output_head(x)
        return x

class SwimUnet(tf.keras.models.Model):
    def __init__(self, num_classes, emb_dims=[96, 192, 384, 768], window_shape=(4, 4, 4), num_heads=16) :
        super().__init__()

        self.encoder = Encoder(emb_dims=emb_dims, window_shape=window_shape, num_heads=num_heads)
        self.decoder = Decoder(emb_dims=emb_dims, window_shape=window_shape, num_heads=num_heads, num_classes=num_classes)

    def call (self, inputs):
        skip=self.encoder(inputs)
        x=self.decoder(skip)
        return x



