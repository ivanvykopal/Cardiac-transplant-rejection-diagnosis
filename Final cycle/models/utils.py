from NestedUnet import NestedUnet
from DeepLabV3Plus import DeepLabV3Plus
from UNet import UNet
from MultiScaleAttentionUnet import MultiScaleAttentionUnet
from StackedUnet import StackedUnet


def get_model(configs):
    if len(configs) == 1 and configs[0].get('model') == 'U-Net++':
        model = NestedUnet(configs[0]).create_model()
        model.load_weights(configs[0]['model_path'])
        return [model]
    elif configs[0].get('model') == 'DeepLabV3+':
        model1 = DeepLabV3Plus(configs[0]).create_model()
        model1.load_weights(configs[0]['model_path'])

        model2 = DeepLabV3Plus(configs[1]).create_model()
        model2.load_weights(configs[1]['model_path'])

        model3 = DeepLabV3Plus(configs[2]).create_model()
        model3.load_weights(configs[2]['model_path'])

        return [model1, model2, model3]
    elif configs[0].get('model') == 'U-Net':
        model = UNet(configs[0]).create_model()
        model.load_weights(configs[0]['model_path'])
        return [model]
    elif configs[0].get('model') == 'MultiScaleAttUnet':
        model = MultiScaleAttentionUnet(configs[0]).create_model()
        model.load_weights(configs[0]['model_path'])
        return [model]
    elif configs[0].get('model') == 'StackedUnet':
        model = StackedUnet(configs[0]).create_model()
        model.load_weights(configs[0]['model_path'])
        return [model]

