from .Nested import NestedUnet
from .CustomNet import CustomNet
from .Unet import Unet
from .AttentionUnet import AttentionUnet
from .DeepLabV3Plus import DeepLabV3Plus
from .CustomNested import CustomNested
from .HRNet import HRNet
from .MultiScaleAttUnet import MultiScaleAttentionUnet
from .StackedUnet import StackedUnet
from .TransUnet import transunet


def get_model(config):
    name = config['model']

    if name.lower() == 'unet':
        return Unet(config).create_model()
    elif name.lower() == 'unet++':
        return NestedUnet(config).create_model()
    elif name.lower() == 'attentionunet':
        return AttentionUnet(config).create_model()
    elif name.lower() == 'deepLabv3+':
        return DeepLabV3Plus(config).create_model()
    elif name.lower() == 'customnested':
        return CustomNested(config).create_model()
    elif name.lower() == 'hrnet':
        return HRNet(config).create_model()
    elif name.lower() in ['multiscaleattunet', 'multiscaleattentionunet']:
        return MultiScaleAttentionUnet(config).create_model()
    elif name.lower() == 'stackedunet':
        return StackedUnet(config).create_model()
    elif name.lower() == 'customnet':
        return CustomNet(config).create_model()
    elif name.lower() == 'transunet':
        return transunet(
            input_size=(512, 512, 3), filter_num=[16, 32, 64, 128], n_labels=3, stack_num_down=2, stack_num_up=2,
            embed_dim=384, num_mlp=768, num_heads=1, num_transformer=1, activation='ReLU', mlp_activation='GELU',
            output_activation='sigmoid', batch_norm=True, pool=True, unpool='bilinear'
        )
    else:
        return Unet(config).create_model()
