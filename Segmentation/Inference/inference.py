import albumentations as A
import argparse
import glob
import json
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from tqdm import tqdm

from Models.models import get_model
from Dataloaders.Datasets import Dataset
from Utils.utils import read_config


def applicate_augmentations(aug, img, mask, batch_size):
    output_img = []
    for image_idx in range(len(img)):
        new_img = np.zeros_like(img[image_idx])
        for idx_batch in range(batch_size):
            augmented = aug(image=img[image_idx][idx_batch], mask=mask[image_idx][idx_batch])
            new_img[idx_batch] = augmented['image']
            mask[image_idx][idx_batch] = augmented['mask']
        output_img.append(new_img)
    return output_img, mask


def inference(model_path, config_name, data_directory='D:/Master Thesis/Code/Segmentation/data4/images',
              output_directory='D:/Master Thesis/CustomNested inference'):
    start = time.time()
    config = read_config(config_name)
    horizontal_flip = A.HorizontalFlip(p=1)
    vertical_flip = A.VerticalFlip(p=1)

    hor_ver_flip = A.Compose([
        A.HorizontalFlip(p=1),
        A.VerticalFlip(p=1)
    ])

    json_path = './data/images_512.json'
    patch_size = config['image_size']

    with open(json_path) as json_file:
        train_valid_json = json.load(json_file)

    model = get_model(config)
    model.load_weights(model_path)

    for image in train_valid_json['images'][0:]:
        file_name = image['name']
        print(file_name)
        files = []
        for idx, fragment in enumerate(image['fragments']):
            img = np.load(f'{data_directory}/{fragment["name"]}.npy')

            fragment_x = fragment['x']
            fragment_y = fragment['y']

            size_x = img.shape[1]
            size_y = img.shape[0]

            x_patches = int(math.ceil(size_x / patch_size))
            y_patches = int(math.ceil(size_y / patch_size))

            for y_patch in range(int(y_patches)):
                for x_patch in range(int(x_patches)):
                    # get x and y coors of the start of the patch
                    x = x_patch * patch_size
                    y = y_patch * patch_size

                    files.append({
                        "patch": [y + fragment_y, x + fragment_x],
                        "name": image['name'],
                        "height": image['height'],
                        "width": image['width'],
                        "fragment": {
                            "idx": idx,
                            "x": fragment['x'],
                            "y": fragment['y'],
                            "width": fragment['width'],
                            "height": fragment['height']
                        },
                        "augmentation": {
                            "type": "normal"
                        }
                    })

        dataset = Dataset(
            batch_size=config['batch_size'],
            directory=data_directory,
            img_json=files,
            config=config,
            in_num=config['input_images'],
            out_num=config['output_masks'],
            inference=True
        )

        index = 0
        for patch in tqdm(dataset, total=len(dataset)):
            img, mask = patch
            pred_mask = model.predict(img)

            img1, mask1 = applicate_augmentations(horizontal_flip, img, mask, config['batch_size'])
            pred_mask1 = model.predict(img1)

            img2, mask2 = applicate_augmentations(vertical_flip, img, mask, config['batch_size'])
            pred_mask2 = model.predict(img2)

            img3, mask3 = applicate_augmentations(hor_ver_flip, img, mask, config['batch_size'])
            pred_mask3 = model.predict(img3)

            pred_mask1, mask1 = applicate_augmentations(horizontal_flip, pred_mask1, mask, config['batch_size'])
            pred_mask2, mask2 = applicate_augmentations(vertical_flip, pred_mask2, mask, config['batch_size'])
            pred_mask3, mask3 = applicate_augmentations(hor_ver_flip, pred_mask3, mask, config['batch_size'])

            if config['model'].lower() == 'stackedunet':
                # treba vymyslieť ako skombinovať aj vysledok zo zvysnych vetiev
                # tento usek kodu odignoruje predikcie pre zvysne vetvy a berie predikcie len pre detail
                pred_mask1 = pred_mask1[0]
                pred_mask2 = pred_mask2[0]
                pred_mask3 = pred_mask3[0]

            final_mask = np.sum([pred_mask, pred_mask1, pred_mask2, pred_mask3], axis=0) / 4

            for idx in range(pred_mask.shape[0]):
                mask = final_mask[idx]

                # závisí či robím multclass alebo multilabel
                directory = files[index]['name']

                if not os.path.isdir(f"{output_directory}/{directory}"):
                    os.mkdir(f"{output_directory}/{directory}")

                np.save(
                    os.path.join(
                        output_directory,
                        directory,
                        f"{files[index]['fragment']['idx']}_{files[index]['patch'][0]}_{files[index]['patch'][1]}.npy"
                    ),
                    mask)
                index += 1

    end = time.time()
    print(f'Overall time: {end - start}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, help='Path to saved model')
    parser.add_argument('--config_name', type=str, help='Config name')
    parser.add_argument('--data_directory', type=str, help='Directory with saved images (fragments)')
    parser.add_argument('--output_directory', type=str, help='Directory to save predictions')

    args = parser.parse_args()
    inference(
        model_path=args.model_path,
        config_name=args.config_name,
        data_directory=args.data_directory,
        output_directory=args.output_directory
    )
