import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, BatchNormalization, UpSampling2D, AveragePooling2D
from tensorflow.keras.layers import MaxPool2D, Activation, Dropout
from tensorflow.keras.layers import Dense, Add, Multiply, concatenate


class BasicConv(Model):
    def __init__(
            self,
            n_filters,
            kernel_size,
            padding='same',
            initializer='he_normal',
            dilation_rate=1,
            batch_norm=False,
            dropout=None,
            residual=False,
            activation='relu'
    ):
        super(BasicConv, self).__init__()

        self.conv = Conv2D(
            n_filters,
            kernel_size=kernel_size,
            padding=padding,
            kernel_initializer=initializer,
            dilation_rate=dilation_rate,
            activation=activation
        )
        self.batch_norm = BatchNormalization() if batch_norm else None
        self.act = Activation(activation)
        self.dropout = Dropout(dropout) if dropout else None
        self.residual = residual

    def call(self, input, training=False):
        x = self.conv(input)
        if self.batch_norm:
            x = self.batch_norm(x, training=training)

        if self.residual:
            x += input

        x = self.act(x)
        if self.dropout:
            x = self.dropout(x)

        return x


class ConvBlock(Model):
    def __init__(self, n_filters, kernel_size=3, stack_num=2, dilation_rate=1, activation='relu', batch_norm=False,
                 residual=False, dropout=None):
        super(ConvBlock, self).__init__()
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.stack_num = stack_num
        self.dilation_rate = dilation_rate
        self.activation = activation
        self.batch_norm = batch_norm
        self.residual = residual
        self.dropout = dropout
        self.convs = [
            BasicConv(
                n_filters=self.n_filters, kernel_size=self.kernel_size, batch_norm=self.batch_norm,
                dropout=self.dropout, residual=self.residual, dilation_rate=self.dilation_rate,
                activation=self.activation
            )
            for _ in range(stack_num)
        ]

    def call(self, x):
        for i in range(self.stack_num):
            x = self.convs[i](x)

        return x


class DownSample(Model):
    def __init__(self, n_filters, stack_num, kernel_size=3, dropout=None, batch_norm=True, pool='max'):
        super(DownSample, self).__init__()
        self.drop = Dropout(dropout) if dropout else None

        self.conv = ConvBlock(
            n_filters=n_filters,
            kernel_size=kernel_size,
            stack_num=stack_num,
            batch_norm=batch_norm,
            dropout=dropout
        )

        if pool == 'max':
            self.pool = MaxPool2D(2)
        elif pool == 'avg':
            self.pool = AveragePooling2D(2)

    def call(self, x):
        f = self.conv(x)
        p = self.pool(f)

        if self.drop:
            p = self.drop(p)
        return f, p


class UpSample(Model):
    def __init__(self, n_filters, kernel_size=3, stack_num=2, padding='same', dropout=None, batch_norm=True, up='conv'):
        super(UpSample, self).__init__()
        self.drop = Dropout(dropout) if dropout else None
        self.conv = ConvBlock(
            n_filters=n_filters,
            kernel_size=kernel_size,
            stack_num=stack_num,
            batch_norm=batch_norm,
            dropout=dropout
        )

        if up == 'conv':
            self.up = Conv2DTranspose(n_filters, 2, 2, padding)
        else:
            self.up = UpSampling2D(size=2, interpolation='bilinear')

    def call(self, x, conv_features):
        x = self.up(x)
        x = concatenate([x, *conv_features])

        if self.drop:
            x = self.drop(x)

        x = self.conv(x)
        return x


class ChannelAttention(Model):
    def __init__(self, features, reduction=16):
        super(ChannelAttention, self).__init__()
        self.features = features
        self.reduction = reduction

        reduced_features = int(features // reduction)
        self.dense1 = Dense(reduced_features)
        self.dense2 = Dense(features)

    def call(self, input):
        avg = tf.reduce_mean(input, axis=[1, 2], keepdims=True)
        max_pool = tf.reduce_max(input, axis=[1, 2], keepdims=True)

        avg_reduced = self.dense1(avg)
        max_pool_reduced = self.dense1(max_pool)

        avg_attention = self.dense2(Activation('relu')(avg_reduced))
        max_pool_attention = self.dense2(Activation('relu')(max_pool_reduced))

        add = Add()([avg_attention, max_pool_attention])
        attention = Activation('sigmoid')(add)
        return input * attention


class SpatialAttention(Model):
    def __init__(self, features, kernel_size):
        super(SpatialAttention, self).__init__()
        self.features = features

        self.conv = Conv2D(
            1,
            kernel_size=kernel_size,
            strides=(1, 1),
            padding='same'
        )

    def call(self, input):
        avg = tf.reduce_mean(input, axis=[3], keepdims=True)  # -1
        max_pool = tf.reduce_max(input, axis=[3], keepdims=True)  # -1

        concat = tf.concat([avg, max_pool], axis=3)
        conv = self.conv(concat)

        attention = Activation('sigmoid')(conv)
        return input * attention


class CBAM(Model):
    def __init__(self, features, kernel_size):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(features)
        self.spatial_attention = SpatialAttention(features, kernel_size)

    def call(self, inputs):
        out = self.channel_attention(inputs)
        out = self.spatial_attention(out)

        return out


class AttentionBlock(Model):
    def __init__(self, n_filters, kernel_size, padding, initializer, batch_norm=False, dropout=None, residual=False):
        super(AttentionBlock, self).__init__()

        self.wx = Conv2D(
            n_filters,
            kernel_size=kernel_size,
            padding=padding,
            kernel_initializer=initializer
        )

        self.wg = Conv2D(
            n_filters,
            kernel_size=kernel_size,
            padding=padding,
            kernel_initializer=initializer
        )

        self.psi = Conv2D(
            n_filters,
            kernel_size=kernel_size,
            padding=padding,
            kernel_initializer=initializer
        )
        self.batch_norm_x = BatchNormalization() if batch_norm else None
        self.batch_norm_g = BatchNormalization() if batch_norm else None
        self.sigmoid = Activation('sigmoid')

    def call(self, x, g):
        x1 = self.wx(x)
        g1 = self.wg(g)
        if self.batch_norm_x and self.batch_norm_g:
            x1 = self.batch_norm_x(x1)
            g1 = self.batch_norm_g(g1)

        activation = Activation('relu')(Add()([x1, g1]))

        out = self.psi(activation)
        out = self.sigmoid(out)
        return Multiply()([x, out])


class UnetBlock(Model):
    def __init__(self, config, num_conv):
        self.config = config
        self.num_conv = num_conv

    def _conv_block(self, x, n_filters, n_convs):
        for i in range(n_convs):
            x = BasicConv(
                n_filters=n_filters,
                kernel_size=self.config['kernel_size'],
                padding=self.config['padding'],
                initializer=self.config['initializer']
            )(x)
        return x

    def _downsample_block(self, x, n_filters, n_convs):
        f = self._conv_block(x, n_filters, n_convs)
        p = MaxPool2D(2)(f)
        p = Dropout(self.config['dropout'])(p)
        return f, p

    def _upsample_block(self, x, conv_features, n_filters, n_convs):
        x = Conv2DTranspose(n_filters, 2, 2, padding=self.config['padding'])(x)
        x = concatenate([x, conv_features])
        x = Dropout(self.config['dropout'])(x)
        x = self._conv_block(x, n_filters, n_convs)
        return x

    def call(self, x):
        f1, p1 = self._downsample_block(
            x, self.config['filters'], self.num_conv)
        # 2 - downsample
        f2, p2 = self._downsample_block(
            p1, self.config['filters'] * 2, self.num_conv)
        # 3 - downsample
        bottleneck = self._conv_block(
            p2, self.config['filters'] * 4, self.num_conv)

        # decoder:
        # 6 - upsample
        u6 = self._upsample_block(
            bottleneck, f2, self.config['filters'] * 2, self.num_conv)
        # 7 - upsample
        u7 = self._upsample_block(
            u6, f1, self.config['filters'], self.num_conv)

        return u7


class DilatedSpatialPyramidPooling(Model):
    def __init__(self, config):
        super(DilatedSpatialPyramidPooling, self).__init__()
        self.config = config

    def call(self, inputs):
        dims = inputs.shape

        x = AveragePooling2D(
            pool_size=(dims[-3], dims[-2])
        )(inputs)

        x = BasicConv(
            n_filters=self.config['filters'],
            kernel_size=1,
            padding=self.config['padding'],
            initializer=self.config['initializer']
        )(x)

        out_pool = UpSampling2D(
            size=(dims[-3] // x.shape[1], dims[-2] // x.shape[2]),
            interpolation="bilinear"
        )(x)

        out_1 = BasicConv(
            n_filters=self.config['filters'],
            kernel_size=1,
            padding=self.config['padding'],
            initializer=self.config['initializer'],
            dilation_rate=1
        )(inputs)
        out_6 = BasicConv(
            n_filters=self.config['filters'],
            kernel_size=3,
            padding=self.config['padding'],
            initializer=self.config['initializer'],
            dilation_rate=6
        )(inputs)
        out_12 = BasicConv(
            n_filters=self.config['filters'],
            kernel_size=3,
            padding=self.config['padding'],
            initializer=self.config['initializer'],
            dilation_rate=12
        )(inputs)
        out_18 = BasicConv(
            n_filters=self.config['filters'],
            kernel_size=3,
            padding=self.config['padding'],
            initializer=self.config['initializer'],
            dilation_rate=18
        )(inputs)

        x = concatenate([out_pool, out_1, out_6, out_12, out_18], axis=-1)
        output = BasicConv(
            n_filters=self.config['filters'],
            kernel_size=1,
            padding=self.config['padding'],
            initializer=self.config['initializer'],
        )(x)

        return output


class ResidualConvBlock(Model):
    def __init__(self, n_filters, kernel_size, padding, initializer):
        super(ResidualConvBlock, self).__init__()
        self.conv1 = Conv2D(
            n_filters,
            kernel_size=kernel_size,
            padding=padding,
            kernel_initializer=initializer
        )
        self.conv2 = Conv2D(
            n_filters,
            kernel_size=kernel_size,
            padding=padding,
            kernel_initializer=initializer
        )
        self.relu = Activation('relu')

    def call(self, inputs):
        x = self.relu(inputs)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)

        return Add()([x, inputs])


class ChainedResidualPooling(Model):
    def __init__(self, n_filters, kernel_size, padding, initializer):
        super(ChainedResidualPooling, self).__init__()
        self.conv1 = Conv2D(
            n_filters,
            kernel_size=kernel_size,
            padding=padding,
            kernel_initializer=initializer
        )
        self.conv2 = Conv2D(
            n_filters,
            kernel_size=kernel_size,
            padding=padding,
            kernel_initializer=initializer
        )
        self.pool1 = MaxPool2D(pool_size=5, strides=1, padding='same')
        self.pool2 = MaxPool2D(pool_size=5, strides=1, padding='same')
        self.relu = Activation('relu')

    def call(self, inputs):
        x1 = self.relu(inputs)
        x = self.pool1(x1)
        x = self.conv1(x)

        add1 = Add()([x, x1])
        x = self.pool2(x)
        x = self.conv2(x)

        return Add()([x, add1])


class MultiResolutionFusion(Model):
    def __init__(self, kernel_size, padding, initializer, n_filters=256):
        super(MultiResolutionFusion, self).__init__()
        self.conv1 = Conv2D(
            n_filters,
            kernel_size=kernel_size,
            padding=padding,
            kernel_initializer=initializer
        )
        self.conv2 = Conv2D(
            n_filters,
            kernel_size=kernel_size,
            padding=padding,
            kernel_initializer=initializer
        )
        self.relu = Activation('relu')

    def call(self, low_inputs=None, high_inputs=None):

        if low_inputs is None:
            return high_inputs

        conv_low = self.conv1(low_inputs)
        conv_high = self.conv2(high_inputs)

        conv_low_up = UpSampling2D(size=2, interpolation='bilinear')(conv_low)

        return Add()([conv_low_up, conv_high])


def CONV_stack(x, channel, kernel_size=3, stack_num=2, dilation_rate=1, activation='ReLU', batch_norm=False):
    bias_flag = not batch_norm

    for i in range(stack_num):
        x = Conv2D(channel, kernel_size=kernel_size, padding='same', use_bias=bias_flag,
                   dilation_rate=dilation_rate)(x)
        if batch_norm:
            x = BatchNormalization()(x)

        act = eval(activation)
        x = act()(x)

    return x


class ConvOut(Model):
    def __init__(self, n_labels, kernel_size, activation):
        super(ConvOut, self).__init__()
        self.activation = activation
        self.conv = Conv2D(n_labels, kernel_size=kernel_size,
                           padding='same', use_bias=True)

    def call(self, inputs):
        x = self.conv(inputs)

        if self.activation:
            x = Activation(self.activation)(x)

        return x


def decode_layer(x, channel, pool_size, unpool, kernel_size=3, activation='ReLU', batch_norm=False):
    if unpool:
        x = UpSampling2D(size=(pool_size, pool_size),
                         interpolation='bilinear')(x)
    else:
        if kernel_size == 'auto':
            kernel_size = pool_size
        x = Conv2DTranspose(channel, kernel_size, strides=(
            pool_size, pool_size), padding='same')(x)
        if batch_norm:
            x = BatchNormalization()(x)

        if activation is not None:
            act = eval(activation)
            x = act()(x)

    return x


def encode_layer(x, channel, pool_size, pool, kernel_size='auto', activation='ReLU', batch_norm=False):
    if pool is True:
        pool = 'max'

    if pool == 'max':
        x = MaxPool2D(pool_size=(pool_size, pool_size))(x)
    elif pool == 'ave':
        x = AveragePooling2D(pool_size=(pool_size, pool_size))(x)
    else:
        if kernel_size == 'auto':
            kernel_size = pool_size

        x = Conv2D(channel, kernel_size, strides=(
            pool_size, pool_size), padding='valid')(x)
        if batch_norm:
            x = BatchNormalization()(x)

        if activation is not None:
            act = eval(activation)
            x = act()(x)

    return x


def Unet_left(x, channel, kernel_size=3, stack_num=2, activation='ReLU', pool=True, batch_norm=False):
    pool_size = 2

    x = encode_layer(x, channel, pool_size, pool,
                     activation=activation, batch_norm=batch_norm)

    x = CONV_stack(x, channel, kernel_size=kernel_size, stack_num=stack_num, activation=activation,
                   batch_norm=batch_norm)
    return x


def Unet_right(x, x_list, channel, kernel_size=3, stack_num=2, activation='ReLU', unpool=True, batch_norm=False,
               concat=True):
    pool_size = 2

    x = decode_layer(x, channel, pool_size, unpool,
                     activation=activation, batch_norm=batch_norm)

    x = CONV_stack(x, channel, kernel_size, stack_num,
                   activation=activation, batch_norm=batch_norm)

    if concat:
        x = concatenate([x, ] + x_list, axis=3)
    x = CONV_stack(x, channel, kernel_size, stack_num=stack_num,
                   activation=activation, batch_norm=batch_norm)

    return x
