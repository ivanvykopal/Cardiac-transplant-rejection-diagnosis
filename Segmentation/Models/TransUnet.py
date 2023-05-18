import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Dense
from tensorflow.keras.layers import add, Conv2D

from Models.Blocks import CONV_stack, ConvOut
from Models.Blocks import Unet_left, Unet_right
from Models.transformer_layers import PatchExtract, PatchEmbedding
from Models.GELU import GELU


def ViT_MLP(x, filter_num, activation='GELU'):

    act = eval(activation)

    for i, f in enumerate(filter_num):
        x = Dense(f)(x)
        x = act()(x)

    return x


def ViT_block(v, num_heads, key_dim, filter_num_MLP, activation='GELU'):
    v_atten = v
    v_atten = LayerNormalization()(v_atten)
    v_atten = MultiHeadAttention(
        num_heads=num_heads, key_dim=key_dim)(v_atten, v_atten)

    v_add = add([v_atten, v])  # skip connection

    v_mlp = v_add
    v_mlp = LayerNormalization()(v_mlp)
    v_mlp = ViT_MLP(v_mlp, filter_num_MLP, activation)

    v_out = add([v_mlp, v_add])

    return v_out


def transunet_base(input, filter_num, stack_num_down, stack_num_up, embed_dim, num_mlp, num_heads, num_transformer,
                   activation, mlp_activation, batch_norm, pool, unpool):

    act = eval(activation)
    x_skip = []
    depth_ = len(filter_num)

    patch_size = 1
    input_size = input.shape[1]
    encode_size = input_size // 2**(depth_ - 1)
    num_patches = encode_size ** 2
    key_dim = embed_dim

    filter_num_MLP = [num_mlp, embed_dim]

    x = input
    x = CONV_stack(x, filter_num[0], stack_num=stack_num_down,
                   activation=activation, batch_norm=batch_norm)
    x_skip.append(x)

    for i, f in enumerate(filter_num[1:]):
        x = Unet_left(x, f, stack_num=stack_num_down,
                      activation=activation, pool=pool, batch_norm=batch_norm)
        x_skip.append(x)

    x = x_skip[-1]
    x_skip = x_skip[:-1]

    x = Conv2D(filter_num[-1], 1, padding='valid', use_bias=False)(x)

    x = PatchExtract((patch_size, patch_size))(x)
    x = PatchEmbedding(num_patches, embed_dim)(x)

    for i in range(num_transformer):
        x = ViT_block(x, num_heads, key_dim, filter_num_MLP,
                      activation=mlp_activation)

    x = tf.reshape(x, (-1, encode_size, encode_size, embed_dim))
    x = Conv2D(filter_num[-1], 1, padding='valid', use_bias=False)(x)

    x_skip.append(x)

    x_skip = x_skip[::-1]
    x = x_skip[0]
    x_decode = x_skip[1:]
    depth_decode = len(x_decode)

    filter_num_decode = filter_num[:-1][::-1]

    for i in range(depth_decode):
        x = Unet_right(x, [x_decode[i], ], filter_num_decode[i], stack_num=stack_num_up, activation=activation,
                       unpool=unpool, batch_norm=batch_norm)

    if depth_decode < depth_ - 1:
        for i in range(depth_ - depth_decode - 1):
            i_real = i + depth_decode
            x = Unet_right(x, None, filter_num_decode[i_real], stack_num=stack_num_up, activation=activation,
                           unpool=unpool, batch_norm=batch_norm, concat=False)

    return x


def transunet(input_size, filter_num, n_labels, stack_num_down, stack_num_up, embed_dim, num_mlp, num_heads,
              num_transformer, activation, mlp_activation, output_activation, batch_norm, pool, unpool):
    input = Input(input_size)

    x = transunet_base(input, filter_num, stack_num_down=stack_num_down, stack_num_up=stack_num_up, embed_dim=embed_dim,
                       num_mlp=num_mlp, num_heads=num_heads, num_transformer=num_transformer, activation=activation,
                       mlp_activation=mlp_activation, batch_norm=batch_norm, pool=pool, unpool=unpool)

    out = ConvOut(n_labels, kernel_size=1, activation=output_activation)(x)

    model = Model(inputs=input, outputs=out)

    return model
