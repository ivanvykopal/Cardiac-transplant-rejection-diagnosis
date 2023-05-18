from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.layers import MaxPooling2D, Dropout, concatenate, Input
from tensorflow.keras import Model
import tensorflow as tf

from .Blocks import BasicConv, CBAM, UnetBlock


class StackedUnet:
    def __init__(self, config):
        super()
        self.config = config

        self.unet2 = UnetBlock(self.config, num_conv=2).create_model(
            (self.config['image_size'] * 2, self.config['image_size'] * 2, self.config['channels']), name='unet2')
        self.unet3 = UnetBlock(self.config, num_conv=2).create_model(
            (self.config['image_size'], self.config['image_size'], self.config['filters'] * 3), name='unet3')

    def _conv_block(self, x, n_filters, n_convs):
        for i in range(n_convs):
            x = BasicConv(
                n_filters=n_filters,
                kernel_size=self.config['kernel_size'],
                padding=self.config['padding'],
                initializer=self.config['initializer'],
                batch_norm=True
            )(x)
        return x

    def _downsample_block(self, x, n_filters, n_convs):
        f = self._conv_block(x, n_filters, n_convs)
        p = MaxPooling2D(2)(f)
        p = Dropout(self.config['dropout'])(p)
        return f, p

    def _upsample_block(self, x, conv_features, n_filters, n_convs):
        x = Conv2DTranspose(n_filters, 2, 2, padding=self.config['padding'])(x)
        x = concatenate([x, conv_features])
        x = Dropout(self.config['dropout'])(x)
        x = self._conv_block(x, n_filters, n_convs)
        return x

    def _cbam_conv_block(self, x, n_filters, cbam_filters):
        x = self._conv_block(x, n_filters, 2)
        x = CBAM(
            features=cbam_filters,
            kernel_size=self.config['kernel_size']
        )(x)
        pool = MaxPooling2D(2)(x)
        return x, pool

    def _cbam_upsample(self, x, conv_features, n_filters):
        x = Conv2DTranspose(n_filters, 2, 2, padding=self.config['padding'])(x)
        x = concatenate([conv_features, x])
        x = self._conv_block(x, n_filters, 2)
        return x

    def create_model(self):
        input1 = Input(
            shape=(self.config['image_size'],
                   self.config['image_size'], self.config['channels'])
        )
        input2 = Input(
            shape=(self.config['image_size'] * 2,
                   self.config['image_size'] * 2, self.config['channels'])
        )

        scale2_unet = self.unet2(input2)
        scale2, pool2 = self._cbam_conv_block(
            input2, self.config['filters'], self.config['filters'])

        input1_conv = self._conv_block(input1, self.config['filters'] * 2, 2)
        scale1 = concatenate([pool2, input1_conv])
        scale1_unet = self.unet3(scale1)
        scale1, pool1 = self._cbam_conv_block(
            scale1, self.config['filters'] * 2, self.config['filters'] * 2)

        scale0, pool0 = self._cbam_conv_block(
            pool1, self.config['filters'] * 4, self.config['filters'] * 4)

        scale, pool = self._cbam_conv_block(
            pool0, self.config['filters'] * 8, self.config['filters'] * 8)

        bottleneck = self._conv_block(pool, self.config['filters'] * 16, 2)

        up_scale0 = self._cbam_upsample(
            bottleneck, scale, self.config['filters'] * 8)

        up_scale1 = self._cbam_upsample(
            up_scale0, scale0, self.config['filters'] * 4)

        up_scale2 = self._cbam_upsample(up_scale1, concatenate(
            [scale1_unet, scale1]), self.config['filters'] * 2)

        up_scale3 = self._cbam_upsample(up_scale2, concatenate(
            [scale2_unet, scale2]), self.config['filters'])

        out_scale1 = Conv2D(
            len(self.config['classes']),
            1,
            padding=self.config['padding'],
            activation=self.config['activation'],
            name='scale1'
        )(up_scale2)

        out_scale2 = Conv2D(
            len(self.config['classes']),
            1,
            padding=self.config['padding'],
            activation=self.config['activation'],
            name='scale2'
        )(up_scale3)

        combined_outputs = concatenate([
            out_scale1,
            tf.keras.layers.Cropping2D(cropping=128)(out_scale2)
        ], name="combined_output")

        model = Model([input1, input2], [out_scale1, out_scale2,
                      combined_outputs], name="StackedU-Net")

        return model
