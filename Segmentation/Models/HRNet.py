import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation
from tensorflow.keras.layers import Input, Add, UpSampling2D
from tensorflow.keras.layers import concatenate
from tensorflow.keras import Model


class HRNet:
    def __init__(self, config):
        super(HRNet, self)
        self.config = config

    def _hr_block(self, x, n_features, stride=(1, 1), shortcut=False):
        out = Conv2D(
            filters=n_features,
            kernel_size=self.config['kernel_size'],
            strides=stride,
            padding=self.config['padding'],
            use_bias=False,
            kernel_initializer=self.config['initializer'],
            bias_initializer='zeros'
        )(x)
        out = BatchNormalization(axis=3)(out)
        out = Activation(tf.keras.activations.relu)(out)
        out = Conv2D(
            filters=n_features,
            kernel_size=self.config['kernel_size'],
            padding=self.config['padding'],
            use_bias=False,
            kernel_initializer=self.config['initializer'],
            bias_initializer='zeros'
        )(out)
        out = BatchNormalization(axis=3)(out)

        if shortcut:
            x = Conv2D(
                filters=n_features,
                kernel_size=1,
                strides=stride,
                use_bias=False,
                kernel_initializer=self.config['initializer'],
                bias_initializer='zeros'
            )(x)
            x = BatchNormalization(axis=3)(x)
            out = Add()([out, x])
        else:
            out = Add()([out, x])

        out = Activation(tf.keras.activations.relu)(out)
        return out

    def _hr_bottleneck_block(self, x, n_feature, stride=(1, 1), shortcut=False, expansion=4):
        n_decode_filter = n_feature // expansion

        out = Conv2D(n_decode_filter, 1, use_bias=False,
                     kernel_initializer=self.config['initializer'], bias_initializer="zeros")(x)
        out = BatchNormalization(axis=3)(out)
        out = Activation(tf.keras.activations.relu)(out)
        out = Conv2D(n_decode_filter, kernel_size=self.config['kernel_size'], strides=stride, padding=self.config['padding'],
                     use_bias=False, kernel_initializer=self.config['initializer'], bias_initializer="zeros")(out)
        out = BatchNormalization(axis=3)(out)
        out = Activation(tf.keras.activations.relu)(out)
        out = Conv2D(n_feature, 1, use_bias=False,
                     kernel_initializer=self.config['initializer'], bias_initializer="zeros")(out)
        out = BatchNormalization(axis=3)(out)

        if shortcut:
            x = Conv2D(n_feature, 1, strides=stride, use_bias=False,
                       kernel_initializer=self.config['initializer'], bias_initializer="zeros")(x)
            x = BatchNormalization(axis=3)(x)
            out = Add()([out, x])
        else:
            out = Add()([out, x])

        out = Activation(tf.keras.activations.relu)(out)
        return out

    def stem_net(self, input):
        x = Conv2D(64, self.config['kernel_size'], strides=(
            2, 2), padding=self.config['padding'], use_bias=False, kernel_initializer=self.config['initializer'])(input)
        x = BatchNormalization(axis=3)(x)
        x = Activation(tf.keras.activations.relu)(x)

        x = self._hr_bottleneck_block(x, 256, shortcut=True)
        x = self._hr_bottleneck_block(x, 256, shortcut=False)
        x = self._hr_bottleneck_block(x, 256, shortcut=False)
        x = self._hr_bottleneck_block(x, 256, shortcut=False)

        return x

    def transition_layer1(self, x, out_filters_list=None):
        if out_filters_list is None:
            out_filters_list = [32, 64]
        x0 = Conv2D(out_filters_list[0], self.config['kernel_size'], padding=self.config['padding'],
                    use_bias=False, kernel_initializer=self.config['initializer'])(x)
        x0 = BatchNormalization(axis=3)(x0)
        x0 = Activation(tf.keras.activations.relu)(x0)

        x1 = Conv2D(out_filters_list[1], 3, strides=(2, 2),
                    padding=self.config['padding'], use_bias=False, kernel_initializer=self.config['padding'])(x)
        x1 = BatchNormalization(axis=3)(x1)
        x1 = Activation(tf.keras.activations.relu)(x1)

        return [x0, x1]

    def make_branch1_0(self, x, out_filters=32):
        x = self._hr_block(x, out_filters, shortcut=False)
        x = self._hr_block(x, out_filters, shortcut=False)
        x = self._hr_block(x, out_filters, shortcut=False)
        x = self._hr_block(x, out_filters, shortcut=False)
        return x

    def make_branch1_1(self, x, out_filters=64):
        x = self._hr_block(x, out_filters, shortcut=False)
        x = self._hr_block(x, out_filters, shortcut=False)
        x = self._hr_block(x, out_filters, shortcut=False)
        x = self._hr_block(x, out_filters, shortcut=False)
        return x

    def fuse_layer1(self, x):
        x0_0 = x[0]
        x0_1 = Conv2D(32, 1, use_bias=False,
                      kernel_initializer='he_normal')(x[1])
        x0_1 = BatchNormalization(axis=3)(x0_1)
        x0_1 = UpSampling2D(size=(2, 2))(x0_1)
        x0 = Add()([x0_0, x0_1])

        x1_0 = Conv2D(64, 3, strides=(2, 2), padding='same',
                      use_bias=False, kernel_initializer='he_normal')(x[0])
        x1_0 = BatchNormalization(axis=3)(x1_0)
        x1_1 = x[1]
        x1 = Add()([x1_0, x1_1])
        return [x0, x1]

    def transition_layer2(self, x, out_filters_list=None):
        if out_filters_list is None:
            out_filters_list = [32, 64, 128]

        x0 = Conv2D(out_filters_list[0], 3, padding='same',
                    use_bias=False, kernel_initializer='he_normal')(x[0])
        x0 = BatchNormalization(axis=3)(x0)
        x0 = Activation('relu')(x0)

        x1 = Conv2D(out_filters_list[1], 3, padding='same',
                    use_bias=False, kernel_initializer='he_normal')(x[1])
        x1 = BatchNormalization(axis=3)(x1)
        x1 = Activation('relu')(x1)

        x2 = Conv2D(out_filters_list[2], 3, strides=(2, 2),
                    padding='same', use_bias=False, kernel_initializer='he_normal')(x[1])
        x2 = BatchNormalization(axis=3)(x2)
        x2 = Activation('relu')(x2)

        return [x0, x1, x2]

    def make_branch2_0(self, x, out_filters=32):
        x = self._hr_block(x, out_filters, shortcut=False)
        x = self._hr_block(x, out_filters, shortcut=False)
        x = self._hr_block(x, out_filters, shortcut=False)
        x = self._hr_block(x, out_filters, shortcut=False)
        return x

    def make_branch2_1(self, x, out_filters=64):
        x = self._hr_block(x, out_filters, shortcut=False)
        x = self._hr_block(x, out_filters, shortcut=False)
        x = self._hr_block(x, out_filters, shortcut=False)
        x = self._hr_block(x, out_filters, shortcut=False)
        return x

    def make_branch2_2(self, x, out_filters=128):
        x = self._hr_block(x, out_filters, shortcut=False)
        x = self._hr_block(x, out_filters, shortcut=False)
        x = self._hr_block(x, out_filters, shortcut=False)
        x = self._hr_block(x, out_filters, shortcut=False)
        return x

    def fuse_layer2(self, x):
        x0_0 = x[0]
        x0_1 = Conv2D(32, 1, use_bias=False,
                      kernel_initializer='he_normal')(x[1])
        x0_1 = BatchNormalization(axis=3)(x0_1)
        x0_1 = UpSampling2D(size=(2, 2))(x0_1)
        x0_2 = Conv2D(32, 1, use_bias=False,
                      kernel_initializer='he_normal')(x[2])
        x0_2 = BatchNormalization(axis=3)(x0_2)
        x0_2 = UpSampling2D(size=(4, 4))(x0_2)
        x0 = Add()([x0_0, x0_1, x0_2])

        x1_0 = Conv2D(64, 3, strides=(2, 2), padding='same',
                      use_bias=False, kernel_initializer='he_normal')(x[0])
        x1_0 = BatchNormalization(axis=3)(x1_0)
        x1_1 = x[1]
        x1_2 = Conv2D(64, 1, use_bias=False,
                      kernel_initializer='he_normal')(x[2])
        x1_2 = BatchNormalization(axis=3)(x1_2)
        x1_2 = UpSampling2D(size=(2, 2))(x1_2)
        x1 = Add()([x1_0, x1_1, x1_2])

        x2_0 = Conv2D(32, 3, strides=(2, 2), padding='same',
                      use_bias=False, kernel_initializer='he_normal')(x[0])
        x2_0 = BatchNormalization(axis=3)(x2_0)
        x2_0 = Activation('relu')(x2_0)
        x2_0 = Conv2D(128, 3, strides=(2, 2), padding='same',
                      use_bias=False, kernel_initializer='he_normal')(x2_0)
        x2_0 = BatchNormalization(axis=3)(x2_0)
        x2_1 = Conv2D(128, 3, strides=(2, 2), padding='same',
                      use_bias=False, kernel_initializer='he_normal')(x[1])
        x2_1 = BatchNormalization(axis=3)(x2_1)
        x2_2 = x[2]
        x2 = Add()([x2_0, x2_1, x2_2])
        return [x0, x1, x2]

    def transition_layer3(self, x, out_filters_list=None):
        if out_filters_list is None:
            out_filters_list = [32, 64, 128, 256]

        x0 = Conv2D(out_filters_list[0], 3, padding='same',
                    use_bias=False, kernel_initializer='he_normal')(x[0])
        x0 = BatchNormalization(axis=3)(x0)
        x0 = Activation('relu')(x0)

        x1 = Conv2D(out_filters_list[1], 3, padding='same',
                    use_bias=False, kernel_initializer='he_normal')(x[1])
        x1 = BatchNormalization(axis=3)(x1)
        x1 = Activation('relu')(x1)

        x2 = Conv2D(out_filters_list[2], 3, padding='same',
                    use_bias=False, kernel_initializer='he_normal')(x[2])
        x2 = BatchNormalization(axis=3)(x2)
        x2 = Activation('relu')(x2)

        x3 = Conv2D(out_filters_list[3], 3, strides=(2, 2),
                    padding='same', use_bias=False, kernel_initializer='he_normal')(x[2])
        x3 = BatchNormalization(axis=3)(x3)
        x3 = Activation('relu')(x3)

        return [x0, x1, x2, x3]

    def make_branch3_0(self, x, out_filters=32):
        x = self._hr_block(x, out_filters, shortcut=False)
        x = self._hr_block(x, out_filters, shortcut=False)
        x = self._hr_block(x, out_filters, shortcut=False)
        x = self._hr_block(x, out_filters, shortcut=False)
        return x

    def make_branch3_1(self, x, out_filters=64):
        x = self._hr_block(x, out_filters, shortcut=False)
        x = self._hr_block(x, out_filters, shortcut=False)
        x = self._hr_block(x, out_filters, shortcut=False)
        x = self._hr_block(x, out_filters, shortcut=False)
        return x

    def make_branch3_2(self, x, out_filters=128):
        x = self._hr_block(x, out_filters, shortcut=False)
        x = self._hr_block(x, out_filters, shortcut=False)
        x = self._hr_block(x, out_filters, shortcut=False)
        x = self._hr_block(x, out_filters, shortcut=False)
        return x

    def make_branch3_3(self, x, out_filters=256):
        x = self._hr_block(x, out_filters, shortcut=False)
        x = self._hr_block(x, out_filters, shortcut=False)
        x = self._hr_block(x, out_filters, shortcut=False)
        x = self._hr_block(x, out_filters, shortcut=False)
        return x

    def fuse_layer3(self, x):
        x0_0 = x[0]
        x0_1 = Conv2D(32, 1, use_bias=False,
                      kernel_initializer='he_normal')(x[1])
        x0_1 = BatchNormalization(axis=3)(x0_1)
        x0_1 = UpSampling2D(size=(2, 2))(x0_1)
        x0_2 = Conv2D(32, 1, use_bias=False,
                      kernel_initializer='he_normal')(x[2])
        x0_2 = BatchNormalization(axis=3)(x0_2)
        x0_2 = UpSampling2D(size=(4, 4))(x0_2)
        x0_3 = Conv2D(32, 1, use_bias=False,
                      kernel_initializer='he_normal')(x[3])
        x0_3 = BatchNormalization(axis=3)(x0_3)
        x0_3 = UpSampling2D(size=(8, 8))(x0_3)
        x0 = concatenate([x0_0, x0_1, x0_2, x0_3], axis=-1)
        return x0

    def final_layer(self, x, classes=1):
        x = UpSampling2D(size=(2, 2))(x)
        x = Conv2D(classes, 1, use_bias=False,
                   kernel_initializer='he_normal')(x)
        x = BatchNormalization(axis=3)(x)
        x = Activation('sigmoid', name='Segmentation')(x)
        return x

    def create_model(self):
        inputs = Input(shape=(
            self.config['image_size'], self.config['image_size'], self.config['channels']))

        x = self.stem_net(inputs)

        x = self.transition_layer1(x)
        x0 = self.make_branch1_0(x[0])
        x1 = self.make_branch1_1(x[1])
        x = self.fuse_layer1([x0, x1])

        x = self.transition_layer2(x)
        x0 = self.make_branch2_0(x[0])
        x1 = self.make_branch2_1(x[1])
        x2 = self.make_branch2_2(x[2])
        x = self.fuse_layer2([x0, x1, x2])

        x = self.transition_layer3(x)
        x0 = self.make_branch3_0(x[0])
        x1 = self.make_branch3_1(x[1])
        x2 = self.make_branch3_2(x[2])
        x3 = self.make_branch3_3(x[3])
        x = self.fuse_layer3([x0, x1, x2, x3])

        out = self.final_layer(x, classes=len(self.config['classes']))
        model = Model(inputs=inputs, outputs=out)
        return model
