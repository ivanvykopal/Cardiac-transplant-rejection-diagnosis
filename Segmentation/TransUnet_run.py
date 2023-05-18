import argparse
import json
import numpy as np
import tensorflow as tf
import wandb

from DataLoaders.IKEMDataset import Dataset
from Models.TransUnet import transunet
from Utils.losses import get_loss
from Utils.metrics import get_metrics
from Train.train import train_cycle
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
    def __init__(self, config, dataloader, num_inputs=1, num_outputs=1):
        super(SemanticLogger, self).__init__()
        self.val_images, self.val_masks = next(iter(dataloader))
        self.config = config
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

    def on_epoch_end(self, logs, epoch):
        pred_masks = self.model.predict(self.val_images)

        if self.num_outputs == 1:
            # Toto len ak v prípade ak ide o class segmentáciu
            pred_masks = np.argmax(pred_masks, axis=-1)
            val_masks = tf.image.convert_image_dtype(self.val_masks, tf.uint8)
            val_masks = np.argmax(val_masks, axis=-1)
            pred_masks = tf.image.convert_image_dtype(pred_masks, tf.uint8)
        else:
            pred_masks[0] = np.argmax(pred_masks[0], axis=-1)
            pred_masks[1] = np.argmax(pred_masks[1], axis=-1)
            pred_masks[2] = np.argmax(pred_masks[2], axis=-1)
            val_masks = [
                tf.image.convert_image_dtype(self.val_masks[0], tf.uint8),
                tf.image.convert_image_dtype(self.val_masks[1], tf.uint8),
                tf.image.convert_image_dtype(self.val_masks[2], tf.uint8)
            ]
            val_masks[0] = np.argmax(val_masks[0], axis=-1)
            val_masks[1] = np.argmax(val_masks[1], axis=-1)
            val_masks[2] = np.argmax(val_masks[2], axis=-1)

            pred_masks[0] = tf.image.convert_image_dtype(
                pred_masks[0], tf.uint8)
            pred_masks[1] = tf.image.convert_image_dtype(
                pred_masks[1], tf.uint8)
            pred_masks[2] = tf.image.convert_image_dtype(
                pred_masks[2], tf.uint8)

        if self.num_inputs == 1:
            val_images = tf.image.convert_image_dtype(
                self.val_images, tf.uint8)
        else:
            val_images = [
                tf.image.convert_image_dtype(self.val_images[0], tf.uint8),
                tf.image.convert_image_dtype(self.val_images[1], tf.uint8),
                tf.image.convert_image_dtype(self.val_images[2], tf.uint8)
            ]

        mask_list = []
        for i in range(len(self.val_images)):
            if self.num_outputs != 1:
                for idx in range(self.num_outputs):
                    mask_list.append(wandb_mask(val_images[idx][i].numpy(),
                                                pred_masks[idx][i].numpy(),
                                                val_masks[idx][i],
                                                self.config))
            else:
                if self.num_inputs != 1 and i == 0:
                    val_images = val_images[0]
                mask_list.append(wandb_mask(val_images[i].numpy(),
                                            pred_masks[i].numpy(),
                                            val_masks[i],
                                            self.config))

        wandb.log({"predictions": mask_list})


def run(wandb_key=None, data_path=None, output_path=None, show_plots=True):
    config_name = 'config_transunet_ikem'
    project_name = 'DP TransU-Net IKEM'

    json_path = './data/images_512-256-splitted.json'
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

    model = transunet(
        input_size=(512, 512, 3), filter_num=[16, 32, 64, 128], n_labels=3, stack_num_down=2, stack_num_up=2,
        embed_dim=384, num_mlp=768, num_heads=1, num_transformer=1, activation='ReLU', mlp_activation='GELU',
        output_activation='sigmoid', batch_norm=True, pool=True, unpool='bilinear'
    )

    with open(json_path) as json_file:
        json_data = json.load(json_file)

    print(f"Train size: {len(json_data['train'])}")

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

    loss = get_loss(config['LOSS']['name'], len(config['FINAL_CLASSES']))
    metrics = get_metrics(config['METRICS'])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=config['LEARNING_RATE']),
        loss=loss,
        metrics=metrics
    )

    saved_model = train_cycle(
        project_name=project_name,
        config=config,
        model=model,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        train_size=len(json_data['train']),
        logger=None,
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
