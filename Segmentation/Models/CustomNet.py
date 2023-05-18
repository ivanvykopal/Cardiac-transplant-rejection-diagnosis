from tensorflow.keras.layers import Input
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D
from .UNetBase import UNetBase


class CustomNet(UNetBase):
    def __init__(self, config):
        super(CustomNet, self).__init__(config)

    def _head(self, bottleneck, skips, name):
        up_scale0 = self._upsample_block(bottleneck, [skips[0]], self.config['filters'] * 8, 2)

        up_scale1 = self._upsample_block(up_scale0, [skips[1]], self.config['filters'] * 4, 2)

        up_scale2 = self._upsample_block(up_scale1, [skips[2]], self.config['filters'] * 2, 2)

        up_scale3 = self._upsample_block(up_scale2, [skips[3]], self.config['filters'], 2)

        return Conv2D(
            1,
            1,
            padding=self.config['padding'],
            activation=self.config['activation'],
            name=name
        )(up_scale3)

    def create_model(self):
        input1 = Input(
            shape=(self.config['image_size'], self.config['image_size'], self.config['channels'])
        )

        scale3, pool3 = self._conv_block(input1, self.config['filters'], 2)

        scale2, pool2 = self._conv_block(pool3, self.config['filters'] * 2, 2)

        scale1, pool1 = self._conv_block(pool2, self.config['filters'] * 4, 2)

        scale0, pool0 = self._conv_block(pool1, self.config['filters'] * 8, 2)

        bottleneck = self._conv_block(pool0, self.config['filters'] * 16, 2)

        head1 = self._head(bottleneck, [scale0, scale1, scale2, scale3], 'blood_vessels')
        head2 = self._head(bottleneck, [scale0, scale1, scale2, scale3], 'inflammations')
        head3 = self._head(bottleneck, [scale0, scale1, scale2, scale3], 'endocariums')

        model = Model(
            input1,
            [head1, head2, head3],
            name="CustomNet"
        )
        return model
