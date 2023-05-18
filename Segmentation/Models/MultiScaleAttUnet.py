from tensorflow.keras.layers import Conv2D, Conv2DTranspose, BatchNormalization, Add, Lambda
from tensorflow.keras.layers import MaxPooling2D, Activation, Dropout, concatenate, Input, multiply, add
from tensorflow.keras.layers import ReLU
from tensorflow.keras import Model
import tensorflow as tf
import tensorflow.keras.backend as K


class MultiScaleAttentionUnet:
    def __init__(self, config):
        super()
        self.config = config

    def _attention_block(self, x, g, inter_channel):
        theta_x = Conv2D(inter_channel, [1, 1], strides=[
                         1, 1], padding=self.config['padding'])(x)
        phi_g = Conv2D(inter_channel, [1, 1], strides=[
                       1, 1], padding=self.config['padding'])(g)

        activation = Activation('relu')(add([theta_x, phi_g]))
        psi_f = Conv2D(1, [1, 1], strides=[1, 1],
                       padding=self.config['padding'])(activation)

        sigmoid = Activation('sigmoid')(psi_f)
        att_x = multiply([x, sigmoid])
        return att_x

    def _attention_up_and_concate(self, x, layer, n_filters):
        in_channel = x.get_shape().as_list()[3]

        up = Conv2DTranspose(
            n_filters, 2, 2, padding=self.config['padding'])(x)
        layer = self._attention_block(
            x=layer, g=up, inter_channel=in_channel // 4)

        my_concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=3))

        concate = my_concat([up, layer])
        return concate

    def _conv_block(self, x, n_filters, n_convs, residual=False):
        out = tf.identity(x)
        for i in range(n_convs):
            out = Conv2D(n_filters, kernel_size=self.config['kernel_size'], padding=self.config['padding'],
                         kernel_initializer=self.config['initializer'])(out)
            out = BatchNormalization()(out)
            out = ReLU()(out)

        if residual:
            shortcut = Conv2D(n_filters, kernel_size=self.config['kernel_size'], padding=self.config['padding'],
                              kernel_initializer=self.config['initializer'])(x)
            shortcut = BatchNormalization()(shortcut)
            out = Add()([shortcut, out])
        return out

    def _downsample_block(self, x, n_filters, n_convs, residual=False):
        f = self._conv_block(x, n_filters, n_convs, residual)
        p = MaxPooling2D(2)(f)
        p = Dropout(self.config['dropout'])(p)
        return f, p

    def _upsample_block(self, x, conv_features, n_filters, n_convs, residual=False):
        x = self._attention_up_and_concate(x, conv_features, n_filters)

        x = Dropout(self.config['dropout'])(x)
        x = self._conv_block(x, n_filters, n_convs, residual)
        return x

    def _upsample_block_basic(self, x, n_filters, n_convs, residual=False):
        x = Conv2DTranspose(n_filters, 2, 2, padding=self.config['padding'])(x)
        x = Dropout(self.config['dropout'])(x)
        x = self._conv_block(x, n_filters, n_convs, residual)
        return x

    def _upsample_block_concat(self, x, conv_features, n_filters, n_convs, residual=False):
        x = Conv2DTranspose(n_filters, 2, 2, padding=self.config['padding'])(x)
        x = concatenate([x, conv_features])
        x = Dropout(self.config['dropout'])(x)
        x = self._conv_block(x, n_filters, n_convs, residual)
        return x

    def create_model(self):
        inputs1 = Input(shape=(
            self.config['image_size'], self.config['image_size'], self.config['channels']))
        inputs2 = Input(shape=(
            self.config['image_size'], self.config['image_size'], self.config['channels']))
        inputs3 = Input(shape=(
            self.config['image_size'], self.config['image_size'], self.config['channels']))

        # handle scale1
        s0 = self._conv_block(inputs1, self.config['filters'], 2)

        # handle scale2
        _, sp1 = self._downsample_block(inputs2, self.config['filters'], 2)
        s1 = self._conv_block(sp1, self.config['filters'] * 2, 2)
        s1 = self._upsample_block_basic(s1, self.config['filters'], 2)

        # handle scale3
        _, sp2 = self._downsample_block(inputs3, self.config['filters'], 2)
        _, sp2 = self._downsample_block(sp2, self.config['filters'] * 2, 2)
        s2 = self._conv_block(sp2, self.config['filters'] * 4, 2)
        s2 = self._upsample_block_basic(s2, self.config['filters'] * 2, 2)
        s2 = self._upsample_block_basic(s2, self.config['filters'], 2)

        scaled = concatenate([s0, s1, s2])

        # encoder
        # 1 - downsample
        f1, p1 = self._downsample_block(scaled, self.config['filters'], 2)
        # 2 - downsample
        f2, p2 = self._downsample_block(p1, self.config['filters'] * 2, 2)
        # 3 - downsample
        f3, p3 = self._downsample_block(p2, self.config['filters'] * 4, 2)
        # 4 - downsample
        f4, p4 = self._downsample_block(p3, self.config['filters'] * 8, 2)
        # 5 - bottleneck
        bottleneck = self._conv_block(p4, self.config['filters'] * 16, 2)

        # decoder:
        # 6 - upsample
        u6 = self._upsample_block(
            bottleneck, f4, self.config['filters'] * 8, 2)
        # 7 - upsample
        u7 = self._upsample_block(u6, f3, self.config['filters'] * 4, 2)
        # 8 - upsample
        u8 = self._upsample_block(u7, f2, self.config['filters'] * 2, 2)
        # 9 - upsample
        u9 = self._upsample_block(u8, f1, self.config['filters'], 2)
        # outputs

        outputs = Conv2D(
            len(self.config['classes']),
            1,
            padding=self.config['padding'],
            activation=self.config['activation']
        )(u9)
        unet_model = Model([inputs1, inputs2, inputs3],
                           outputs, name="MultiScaleAttentionUNet")

        return unet_model
