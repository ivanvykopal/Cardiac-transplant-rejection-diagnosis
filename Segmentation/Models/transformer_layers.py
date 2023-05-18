import tensorflow as tf
from tensorflow.image import extract_patches
from tensorflow.keras.layers import Layer, Dense, Embedding


class PatchExtract(Layer):
    def __init__(self, patch_size, **kwargs):
        super(PatchExtract, self).__init__(**kwargs)
        self.patch_size = patch_size
        self.patch_size_x = patch_size[0]
        self.patch_size_y = patch_size[0]

    def call(self, images):
        batch_size = tf.shape(images)[0]

        patches = extract_patches(
            images=images,
            sizes=(1, self.patch_size_x, self.patch_size_y, 1),
            strides=(1, self.patch_size_x, self.patch_size_y, 1),
            rates=(1, 1, 1, 1),
            padding='VALID'
        )

        patch_dim = patches.shape[-1]
        patch_num = patches.shape[1]
        patches = tf.reshape(
            patches, (batch_size, patch_num * patch_num, patch_dim))

        return patches


class PatchEmbedding(Layer):
    def __init__(self, num_patch, embed_dim, **kwargs):
        super(PatchEmbedding, self).__init__(**kwargs)
        self.num_patch = num_patch
        self.embed_dim = embed_dim
        self.proj = Dense(embed_dim)
        self.pos_embed = Embedding(input_dim=num_patch, output_dim=embed_dim)

    def call(self, patch):
        pos = tf.range(start=0, limit=self.num_patch, delta=1)

        embed = self.proj(patch) + self.pos_embed(pos)
        return embed
