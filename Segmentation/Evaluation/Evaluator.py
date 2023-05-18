import numpy as np
import tensorflow as tf
from Utils.eval import display_sample
from .utils import create_prediction_img, create_RGB_mask, get_labels


class Evaluator:
    def __init__(self, model, model_path, config, out_dir, show_plots=False):
        model.load_weights(f'{out_dir}/{model_path}')
        self.model = model
        self.batch_size = config['batch_size']
        self.classes = config['classes']
        self.num_classes = len(self.classes)
        self.show_plots = show_plots
        self.lab = config['lab']
        self.image_size = config['image_size']
        self.config = config

        self.in_num = config['input_images']
        self.out_num = config['output_masks']

        self.multiclass = config['activation'] == 'softmax'

    def evaluate(self, datasets, num_batches=2):
        datasets_labels = ['TRAIN', 'VALID', 'TEST']
        for dataset_idx, dataset in enumerate(datasets):
            print(f'{datasets_labels[dataset_idx]} DATASET')
            for idx_batch, patch in enumerate(dataset):
                if idx_batch == num_batches:
                    break

                img, mask = patch
                pred_mask = self.model.predict(img)
                for index_img in range(self.batch_size):
                    if self.in_num == 1 and self.out_num == 1:
                        self._eval_in1_out1(img, mask, pred_mask, index_img)
                    elif self.in_num == 1 and self.out_num == 3:
                        self._eval_in1_out3(img, mask, pred_mask, index_img)
                    else:
                        self._eval_inX_outX(img, mask, pred_mask, index_img)

        results = self.model.evaluate(datasets[2], batch_size=self.batch_size)
        print(results)

    def _eval_in1_out1(self, img, mask, pred_mask, index_img):
        images = [img[index_img][:, :, :3]]
        labels_caption = ['Original image']
        if self.multiclass:
            labels = get_labels()
            ground_truth = mask[index_img][:, :, 0]
            predicted = np.argmax(tf.nn.softmax(pred_mask[index_img]), axis=-1)
            images = [
                *images,
                create_RGB_mask(ground_truth, labels, self.image_size),
                create_RGB_mask(predicted, labels, self.image_size),
            ]

            labels_caption = [
                *labels_caption,
                'Ground truth',
                'Prediction',
            ]
        else:
            ground_truth = mask[index_img]
            predicted = pred_mask[index_img]
            for idx in range(self.num_classes):
                images.append(create_prediction_img(
                    ground_truth[:, :, idx], predicted[:, :, idx]))
            labels_caption = [
                'Original image',
                *[f'Prediction - {c}' for c in self.classes]
            ]

        if self.show_plots:
            display_sample(images, labels_caption)

    def _eval_inX_outX(self, img, mask, pred_mask, index_img):
        if self.lab:
            images = [img[idx][index_img][:, :, :3]
                      for idx in range(self.in_num)]
        else:
            images = [img[idx][index_img][:, :, :3]
                      for idx in range(self.in_num)]

        if self.out_num == 1:
            for idx in range(self.num_classes):
                images.append(create_prediction_img(
                    mask[index_img][:, :, idx], pred_mask[index_img][:, :, idx]))

            labels = [
                *[f'Image - size {idx}' for idx in range(self.in_num)],
                *[f'{c}' for c in self.classes]
            ]
            display_sample(images, labels)
        else:
            for scale in range(self.in_num):
                for idx in range(self.num_classes):
                    images.append(
                        create_prediction_img(
                            mask[scale][index_img][:, :, idx], pred_mask[scale][index_img][:, :, idx])
                    )
            labels = [
                *[f'Scale{idx + 1} image' for idx in range(self.in_num)],
                *[
                    f'Scale{idx + 1} {c}'
                    for c in self.classes
                    for idx in range(self.in_num)
                ]
            ]

            if self.show_plots:
                display_sample(images[:self.in_num], labels[:self.in_num])
                display_sample(
                    images[self.in_num:self.in_num + self.num_classes],
                    labels[self.in_num:self.in_num + self.num_classes]
                )
                display_sample(
                    images[self.in_num +
                           self.num_classes:self.in_num + 2*self.num_classes],
                    labels[self.in_num +
                           self.num_classes:self.in_num + 2*self.num_classes]
                )
                if self.in_num == 3:
                    display_sample(
                        images[self.in_num + 2*self.num_classes:],
                        labels[self.in_num + 2*self.num_classes:]
                    )

    def _eval_in1_out3(self, img, mask, pred_mask, index_img):
        images = [img[index_img[:, :, :3]]]

        for idx in range(self.num_classes):
            images.append(create_prediction_img(
                mask[idx][:, :, 0], pred_mask[idx][:, :, 0]))

        labels_caption = [
            'Original image',
            *[f'Prediction - {c}' for c in self.classes]
        ]

        if self.show_plots:
            display_sample(images, labels_caption)
