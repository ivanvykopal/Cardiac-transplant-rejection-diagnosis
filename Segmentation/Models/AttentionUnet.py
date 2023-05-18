from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.layers import Activation, Lambda
from tensorflow.keras.layers import Input, multiply, add
from tensorflow.keras import Model
from tensorflow.keras import backend as K

from .UNetBase import UNetBase
from .Blocks import ConvOut


class AttentionUnet(UNetBase):
    def __init__(self, config):
        super(AttentionUnet, self).__init__(config)

    def _attention_block(self, x, g, inter_channel):
        theta_x = Conv2D(inter_channel, kernel_size=1,
                         strides=1, padding=self.config['padding'])(x)
        phi_g = Conv2D(inter_channel, kernel_size=1, strides=1,
                       padding=self.config['padding'])(g)

        activation = Activation('relu')(add([theta_x, phi_g]))
        psi_f = Conv2D(1, kernel_size=1, strides=1,
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

    def _upsample_block(self, x, conv_features, n_filters, n_convs):
        x = self._attention_up_and_concate(x, conv_features, n_filters)

        # x = Dropout(self.config['DROPOUT'])(x)
        x = self._conv_block(x, n_filters, n_convs)
        return x

    def create_model(self):
        inputs = Input(shape=(
            self.config['image_size'], self.config['image_size'], self.config['channels']))
        # encoder
        # 1 - downsample
        f1, p1 = self._downsample_block(inputs, self.config['filters'], 2)
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
        outputs = ConvOut(
            n_labels=len(self.config['classes']),
            kernel_size=1,
            activation=self.config['activation']
        )(u9)

        unet_model = Model(inputs, outputs, name="Attention U-Net")

        return unet_model
