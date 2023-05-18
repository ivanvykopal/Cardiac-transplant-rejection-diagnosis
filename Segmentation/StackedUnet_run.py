import argparse
import json
import cv2 as cv
import numpy as np
import tensorflow as tf
import wandb
import os

from DataLoaders.Dataset import Dataset
from Models.StackedUnet import StackedUnet
from Utils.losses import get_loss
from Utils.metrics import get_metrics
from Train.train import train_cycle
from Utils.data_split import train_test_split_fragments
from Evaluation.Evaluator import Evaluator


def labels(config):
    l = {}
    for i, label in enumerate(config['FINAL_CLASSES']):
        l[i] = label
    return l


# util function for generating interactive image mask from components
def wandb_mask(bg_img, pred_mask, true_mask, config):
    if true_mask.ndim == 3:
        true_mask = true_mask[:, :, 0]

    return wandb.Image(bg_img, masks={
        "prediction": {
            "mask_data": pred_mask,
            "class_labels": labels(config)
        },
        "ground truth": {
            "mask_data": true_mask,
            "class_labels": labels(config)
        }
    }
    )


class SemanticLogger(tf.keras.callbacks.Callback):
    def __init__(self, config, dataloader, output_dir, num_inputs=1, num_outputs=1):
        super(SemanticLogger, self).__init__()
        self.val_images, self.val_masks = next(iter(dataloader))
        self.config = config
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.output_dir = output_dir

    def on_epoch_end(self, epoch, logs):
        os.mkdir(f"{self.output_dir}/{epoch}")
        pred_masks = self.model.predict(self.val_images)

        mask_captions = self.config['CLASSES']

        for i in range(len(self.val_images)):
            image = self.val_images[i][0]
            cv.imwrite(f"{self.output_dir}/{epoch}/Scale {i}.png", image * 255)

            for c_idx, c in enumerate(mask_captions):
                mask = self.val_masks[i][0][:, :, c_idx]
                truth_mask = np.zeros((mask.shape[0], mask.shape[1], 3))
                truth_mask[:, :, 0] = mask
                truth_mask[:, :, 1] = mask
                truth_mask[:, :, 2] = mask
                cv.imwrite(
                    f"{self.output_dir}/{epoch}/TRUTH Scale {i} {c}.png", truth_mask * 255)

                mask = pred_masks[i][0][:, :, c_idx] > self.config['THRESHOLD']
                pred_mask = np.zeros((mask.shape[0], mask.shape[1], 3))
                pred_mask[:, :, 0] = mask
                pred_mask[:, :, 1] = mask
                pred_mask[:, :, 2] = mask
                cv.imwrite(
                    f"{self.output_dir}/{epoch}/PRED Scale {i} {c}.png", pred_mask * 255)


def run(wandb_key=None, data_path=None, output_path=None, show_plots=True):
    config_name = 'config_stacked_ikem'
    project_name = 'DP StackedUnet IKEM - Azure'
    json_path = './data/images_256.json'
    if data_path is None:
        data_directory = 'D:\\Master Thesis\\Code\\Segmentation\\data4'
    else:
        data_directory = f"{data_path}"
    print(data_directory)

    if output_path is None:
        output_path = 'D:/Master Thesis/Code/Segmentation/results'

    f = open('Configs/' + str(config_name) + '.json')
    config = json.load(f)
    f.close()

    model = StackedUnet(config).create_model()

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

    metrics = get_metrics(config['METRICS'])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=config['LEARNING_RATE']),
        loss={
            'scale1': get_loss(config['LOSS']['name'], len(config['CLASSES'])),
            'scale2': get_loss(config['LOSS']['name'], len(config['CLASSES'])),
            'scale3': get_loss(config['LOSS']['name'], len(config['CLASSES']))
        },
        loss_weights={'scale1': 1.0, 'scale2': 0.75, 'scale3': 0.5},
        metrics=metrics
    )

    saved_model = train_cycle(
        project_name=project_name,
        config=config,
        model=model,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        train_size=train_size,
        logger=SemanticLogger(config, train_dataset, output_dir=output_path),
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

    args = parser.parse_args()
    run(wandb_key=args.wandb, data_path=args.data_path,
        output_path=args.output_path, show_plots=False)
