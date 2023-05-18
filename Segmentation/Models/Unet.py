from tensorflow.keras.layers import Conv2D, Conv2DTranspose, BatchNormalization, MaxPool2D
from tensorflow.keras.layers import Activation, Dropout, concatenate, Input, UpSampling2D, AveragePooling2D
from tensorflow_addons.layers import GroupNormalization
from tensorflow.keras import Model
from .Blocks import ConvOut


class UNet:
    def __init__(self, config):
        self.config = config

    def _conv_block(self, x, n_filters, n_convs):
        for i in range(n_convs):
            x = Conv2D(n_filters, kernel_size=self.config['kernel_size'], padding=self.config['padding'],
                       kernel_initializer=self.config['initializer'])(x)
            #x = BatchNormalization()(x)
            x = GroupNormalization(groups=self.config['batch_size'])(x)
            x = Activation('relu')(x)

        return x

    def _downsample_block(self, x, n_filters, n_convs):
        f = self._conv_block(x, n_filters, n_convs)
        if self.config['pool'] == 'max':
            p = MaxPool2D(2)(f)
        elif self.config['pool'] == 'avg':
            p = AveragePooling2D(2)(f)
        p = Dropout(self.config['dropout'])(p)
        return f, p

    def _upsample_block(self, x, conv_features, n_filters, n_convs):
        if self.config['up'] == 'conv':
            x = Conv2DTranspose(
                n_filters, 2, 2, padding=self.config['padding'])(x)
        else:
            x = UpSampling2D(size=2, interpolation='bilinear')(x)
        x = concatenate([x, *conv_features])
        x = Dropout(self.config['dropout'])(x)
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
            bottleneck, [f4], self.config['filters'] * 8, 2)
        # 7 - upsample
        u7 = self._upsample_block(u6, [f3], self.config['filters'] * 4, 2)
        # 8 - upsample
        u8 = self._upsample_block(u7, [f2], self.config['filters'] * 2, 2)
        # 9 - upsample
        u9 = self._upsample_block(u8, [f1], self.config['filters'], 2)

        # outputs
        outputs = ConvOut(
            n_labels=len(self.config['classes']),
            kernel_size=1,
            activation=self.config['activation']
        )(u9)
        unet_model = Model(inputs, outputs, name="U-Net")

        return unet_model
