from tensorflow.keras.layers import Input, Add
from tensorflow.keras import Model

from .UNetBase import UNetBase
from .Blocks import ConvOut


class NestedUnet(UNetBase):
    def __init__(self, config):
        super(NestedUnet, self).__init__(config)

    def create_model(self):
        inputs = Input(shape=(self.config['image_size'], self.config['image_size'], self.config['channels']))
        # encoder
        # 1 - downsample
        conv1_1, pool1 = self._downsample_block(inputs, self.config['filters'], 2)
        # 2 - downsample
        conv2_1, pool2 = self._downsample_block(pool1, self.config['filters'] * 2, 2)

        conv1_2 = self._upsample_block(conv2_1, [conv1_1], self.config['filters'], 2)

        # 3 - downsample
        conv3_1, pool3 = self._downsample_block(pool2, self.config['filters'] * 4, 2)

        conv2_2 = self._upsample_block(conv3_1, [conv2_1], self.config['filters'] * 2, 2)
        conv1_3 = self._upsample_block(conv2_2, [conv1_1, conv1_2], self.config['filters'], 2)

        # 4 - downsample
        conv4_1, pool4 = self._downsample_block(pool3, self.config['filters'] * 8, 2)

        conv3_2 = self._upsample_block(conv4_1, [conv3_1], self.config['filters'] * 4, 2)
        conv2_3 = self._upsample_block(conv3_2, [conv2_1, conv2_2], self.config['filters'] * 2, 2)
        conv1_4 = self._upsample_block(conv2_3, [conv1_1, conv1_2, conv1_3], self.config['filters'], 2)

        # 5 - bottleneck
        conv5_1 = self._conv_block(pool4, self.config['filters'] * 16, 2)

        conv4_2 = self._upsample_block(conv5_1, [conv4_1], self.config['filters'] * 8, 2)
        conv3_3 = self._upsample_block(conv4_2, [conv3_1, conv3_2], self.config['filters'] * 4, 2)
        conv2_4 = self._upsample_block(conv3_3, [conv2_1, conv2_2, conv2_3], self.config['filters'] * 2, 2)
        conv1_5 = self._upsample_block(conv2_4, [conv1_1, conv1_2, conv1_3, conv1_4], self.config['filters'], 2)

        # outputs
        outputs = ConvOut(
            n_labels=len(self.config['classes']),
            kernel_size=1,
            activation=self.config['activation']
        )(conv1_5)

        nested_model = Model(inputs, outputs, name="NestedUNet")

        return nested_model
