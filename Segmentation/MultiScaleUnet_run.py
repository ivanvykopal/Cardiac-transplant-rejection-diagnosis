import argparse
import json
import numpy as np
import tensorflow as tf
import wandb

from DataLoaders.Dataset import Dataset
from Models.MultiScaleAttUnet import MultiScaleAttentionUnet
from Utils.optimizer import get_optimizer
from Train.train import train_cycle
from Utils.losses import get_loss
from Utils.metrics import get_metrics
from Evaluation.Evaluator import Evaluator


def labels(config):
    l = {}
    for i, label in enumerate(config['FINAL_CLASSES']):
        l[i] = label
    return l


# util function for generating interactive image mask from components
def wandb_mask(img, caption):
    return wandb.Image(img, caption=caption)


def create_prediction_img(pred, true):
    new_image = np.zeros((pred.shape[0], pred.shape[1], 3))
    new_image[:, :, 0] = true
    new_image[:, :, 1] = pred

    return new_image


class SemanticLogger(tf.keras.callbacks.Callback):
    def __init__(self, config, dataloader, num_inputs=1, num_outputs=1):
        super(SemanticLogger, self).__init__()
        self.val_images, self.val_masks = next(iter(dataloader))
        self.config = config
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

    def on_epoch_end(self, epoch, logs):
        pred_masks = self.model.predict(self.val_images)
        val_masks = [
            tf.image.convert_image_dtype(self.val_masks[0], tf.uint8),
            tf.image.convert_image_dtype(self.val_masks[1], tf.uint8),
            tf.image.convert_image_dtype(self.val_masks[2], tf.uint8)
        ]
        val_images = [
            tf.image.convert_image_dtype(self.val_images[0], tf.uint8),
            tf.image.convert_image_dtype(self.val_images[1], tf.uint8),
            tf.image.convert_image_dtype(self.val_images[2], tf.uint8)
        ]
        pred_masks = tf.image.convert_image_dtype(pred_masks, tf.uint8)

        mask_list = list()

        for i in range(len(val_masks)):
            for scale in range(3):
                mask_list.append(wandb_mask(
                    val_images[scale][i].numpy(), caption=f'Image {i} - scale {scale + 1}'))

            for idx_c, c in enumerate(self.config['CLASSES']):
                mask_list.append(
                    wandb_mask(
                        create_prediction_img(pred_masks[i].numpy()[
                                              :, :, idx_c], val_masks[i].numpy()[:, :, idx_c]),
                        caption=f'Image {i} - {c}'
                    )
                )

        wandb.log({"Prediction": mask_list})


def run(project_name, config_name, wandb_key=None, data_path=None, output_path=None, show_plots=True):
    f = open('Configs/' + str(config_name) + '.json')
    config = json.load(f)
    f.close()

    json_path = config['JSON_PATH']

    if data_path is None:
        data_directory = config['DATA_PATH']
    else:
        data_directory = f"{data_path}"
    print(data_directory)

    if output_path is None:
        output_path = config['OUTPUT_PATH']

    model = MultiScaleAttentionUnet(config).create_model()

    with open(json_path) as json_file:
        json_data = json.load(json_file)

    train_size = len(json_data['train']) * (len(config['AUGMENTATIONS']) + 1)
    print(f"Train size: {train_size}")

    train_dataset = Dataset(
        batch_size=config['BATCH_SIZE'],
        directory=data_directory,
        img_json=json_data['train'],
        config=config,
        augmentations=config['AUGMENTATIONS']
    )
    valid_dataset = Dataset(
        batch_size=config['BATCH_SIZE'],
        directory=data_directory,
        img_json=json_data['valid'],
        config=config,
        augmentations=[]
    )
    test_dataset = Dataset(
        batch_size=config['BATCH_SIZE'],
        directory=data_directory,
        img_json=json_data['test'],
        config=config,
        augmentations=[]
    )

    loss = get_loss(config['LOSS']['name'], len(config['CLASSES']))
    metrics = get_metrics(config['METRICS'])
    optimizer = get_optimizer(config['LEARNING_RATE'], config['OPTIMIZER'])

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )

    saved_model = train_cycle(
        project_name=project_name,
        config=config,
        model=model,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        train_size=train_size,
        logger=SemanticLogger(config, train_dataset,
                              num_inputs=3, num_outputs=1),
        wandb_key=wandb_key,
        output_path=output_path
    )

    Evaluator(
        model=model,
        model_path=saved_model,
        config=config,
        out_dir=output_path,
        show_plots=show_plots
    ).evaluate(datasets=[train_dataset, valid_dataset, test_dataset])


# Note: Please run Run.ipynb instead
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb', type=str, help='wandb API token')
    parser.add_argument('--data_path', type=str, help='path to data')
    parser.add_argument('--output_path', type=str, help='path to output')
    parser.add_argument('--project_name', type=str,
                        help='name for wandb project', required=True)
    parser.add_argument(
        '--config_name',
        type=str,
        help='name of the config file',
        default='new_config',
        required=True
    )

    args = parser.parse_args()
    run(
        project_name=args.project_name,
        config_name=args.config_name,
        wandb_key=args.wandb,
        data_path=args.data_path,
        output_path=args.output_path,
        show_plots=False
    )
