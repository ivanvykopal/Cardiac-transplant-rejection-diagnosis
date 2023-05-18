from .Blocks import ConvBlock, DownSample, UpSample


class UNetBase:
    def __init__(self, config):
        super().__init__()
        self.config = config

    def _conv_block(self, x, n_filters, n_convs):
        x = ConvBlock(
            n_filters=n_filters,
            kernel_size=self.config['kernel_size'],
            stack_num=n_convs,
            dropout=self.config['dropout']
        )(x)
        return x

    def _downsample_block(self, x, n_filters, n_convs):
        f, p = DownSample(
            n_filters=n_filters,
            stack_num=n_convs,
            kernel_size=self.config['kernel_size'],
            dropout=self.config['dropout'],
            pool=self.config['pool']
        )(x)
        return f, p

    def _upsample_block(self, x, conv_features, n_filters, n_convs):
        x = UpSample(
            n_filters=n_filters,
            kernel_size=self.config['kernel_size'],
            stack_num=n_convs,
            dropout=self.config['dropout'],
            up=self.config['up']
        )(x, conv_features)
        return x

    def create_model(self):
        raise NotImplementedError
