import tensorflow as tf
from tensorflow.python.ops import bitwise_ops
import wandb

from Evaluation.utils import create_prediction_img, create_RGB_mask, get_labels


# util function for generating interactive image mask from components
def wandb_mask(img, caption):
    return wandb.Image(img, caption=caption)


def convert_2_multiclass(mask, num_classes):
    union = mask[Ellipsis, 0]
    for i in range(1, num_classes):
        union = bitwise_ops.bitwise_or(union, mask[Ellipsis, i])
    union = tf.clip_by_value(union, clip_value_min=0, clip_value_max=1)
    background = tf.where((union == 0) | (union == 1), union ^ 1, union)

    return tf.concat([tf.expand_dims(background, axis=-1), mask], axis=-1)


def combined_output(y_true, y_pred, num_classes):
    mask1 = tf.cast(y_pred[Ellipsis, :num_classes] > 0.5, tf.int32)
    mask2 = tf.cast(y_pred[Ellipsis, num_classes:2 *
                    num_classes] > 0.5, tf.int32)
    mask3 = tf.cast(y_pred[Ellipsis, 2*num_classes:] > 0.5, tf.int32)

    true_mask = y_true[Ellipsis, :num_classes]
    true_mask = convert_2_multiclass(tf.cast(true_mask, tf.int32), num_classes)

    intersect = tf.bitwise.bitwise_and(
        tf.bitwise.bitwise_and(mask1, mask2), mask3)
    intersect = convert_2_multiclass(intersect, num_classes)

    intersect_max = tf.argmax(intersect, axis=-1)
    intersect_max = tf.expand_dims(intersect_max, -1)

    true_max = tf.argmax(true_mask, axis=-1)
    true_max = tf.expand_dims(true_max, axis=-1)

    return true_max, intersect_max


class Logger(tf.keras.callbacks.Callback):
    def __init__(self, config, dataloader):
        super(Logger, self).__init__()
        self.val_images, self.val_masks = next(iter(dataloader))
        self.config = config
        self.num_inputs = config['input_images']
        self.num_outputs = config['output_masks']
        self.labels = get_labels()

        if self.num_outputs == 1:
            if self.num_inputs == 1:
                self.eval = self._in1_out1
            elif self.num_inputs == 2:
                self.eval = self._in2_out1
            else:
                self.eval = self._in3_out1
        elif self.num_outputs == 3 and self.num_inputs == 2:
            self.eval = self._in2_out3
        elif self.num_outputs == 3 and self.num_inputs == 1:
            self.eval = self._in1_out3
        elif self.num_outputs == 4 and self.num_inputs == 3:
            self.eval = self._in3_out4
        else:
            raise ValueError

    def on_epoch_end(self, epoch, logs):
        pred_masks = self.model.predict(self.val_images)

        mask_list = self.eval(pred_masks)

        wandb.log({"Prediction": mask_list})

    def _in1_out1(self, pred_masks):
        return []

    def _in1_out3(self, pred_masks):
        val_masks = [
            tf.image.convert_image_dtype(self.val_masks[0], tf.uint8),
            tf.image.convert_image_dtype(self.val_masks[1], tf.uint8),
            tf.image.convert_image_dtype(self.val_masks[2], tf.uint8)
        ]
        val_images = tf.image.convert_image_dtype(self.val_images, tf.uint8)
        pred_masks = [
            tf.image.convert_image_dtype(pred_masks[0], tf.uint8),
            tf.image.convert_image_dtype(pred_masks[1], tf.uint8),
            tf.image.convert_image_dtype(pred_masks[2], tf.uint8)
        ]

        mask_list = list()
        for i in range(len(self.val_images)):
            mask_list.append(
                wandb_mask(
                    val_images[i].numpy()[:, :, :3],
                    caption=f'Image {i}'
                )
            )
            for idx_c, c in enumerate(self.config['classes']):
                mask_list.append(
                    wandb_mask(
                        create_prediction_img(
                            val_masks[i][idx_c].numpy()[:, :, 0],
                            pred_masks[i][idx_c].numpy()[:, :, 0]
                        ),
                        caption=f'Image {i} - {c}'
                    )
                )

        return mask_list

    def _in2_out1(self, pred_masks):
        val_mask = tf.image.convert_image_dtype(self.val_masks, tf.uint8)
        val_images = [
            tf.image.convert_image_dtype(self.val_images[0], tf.uint8),
            tf.image.convert_image_dtype(self.val_images[1], tf.uint8),
        ]
        pred_mask = tf.image.convert_image_dtype(pred_masks, tf.uint8)

        mask_list = list()

        for batch_idx in range(self.config['batch_size']):
            mask_list.append(wandb_mask(val_images[0].numpy()[
                             batch_idx, :, :, :3], caption=f'Image - size 1'))
            mask_list.append(wandb_mask(val_images[1].numpy()[
                             batch_idx, :, :, :3], caption=f'Image - size 2'))

            for idx_c, c in enumerate(self.config['classes']):
                mask_list.append(
                    wandb_mask(
                        create_prediction_img(val_mask.numpy()[batch_idx, :, :, idx_c], pred_mask.numpy()[
                                              batch_idx, :, :, idx_c]),
                        caption=f'Prediction - {c}'
                    )
                )

        return mask_list

    def _in3_out1(self, pred_masks):
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
                mask_list.append(wandb_mask(val_images[scale][i][:, :, :3].numpy(
                ), caption=f'Image {i} - scale {scale + 1}'))

            for idx_c, c in enumerate(self.config['classes']):
                mask_list.append(
                    wandb_mask(
                        create_prediction_img(pred_masks[i].numpy()[
                                              :, :, idx_c], val_masks[i].numpy()[:, :, idx_c]),
                        caption=f'Image {i} - {c}'
                    )
                )

        return mask_list

    def _in2_out3(self, pred_masks):
        val_masks = [
            tf.image.convert_image_dtype(self.val_masks[0], tf.uint8),
            tf.image.convert_image_dtype(self.val_masks[1], tf.uint8),
            tf.image.convert_image_dtype(self.val_masks[2], tf.uint8)
        ]
        val_images = [
            tf.image.convert_image_dtype(self.val_images[0], tf.uint8),
            tf.image.convert_image_dtype(self.val_images[1], tf.uint8)
        ]
        pred_masks = [
            tf.image.convert_image_dtype(pred_masks[0], tf.uint8),
            tf.image.convert_image_dtype(pred_masks[1], tf.uint8),
            tf.image.convert_image_dtype(pred_masks[2], tf.uint8)
        ]

        mask_list = list()
        for i in range(len(self.val_images)):
            for scale in range(2):
                mask_list.append(
                    wandb_mask(
                        val_images[scale][i].numpy()[:, :, :3],
                        caption=f'Image {i} - scale {scale + 1}'
                    )
                )
                for idx_c, c in enumerate(self.config['classes']):
                    mask_list.append(
                        wandb_mask(
                            create_prediction_img(val_masks[scale][i].numpy()[
                                                  :, :, idx_c], pred_masks[scale][i].numpy()[:, :, idx_c]),
                            caption=f'Image {i} - {c}'
                        )
                    )

            true, predicted = combined_output(val_masks[2][i].numpy(
            ), pred_masks[2][i].numpy(), len(self.config['classes']))
            mask_list.append(
                wandb_mask(
                    create_RGB_mask(true[:, :, 0], self.labels, self.config),
                    caption=f'Combined output true {i}'
                )
            )

            mask_list.append(
                wandb_mask(
                    create_RGB_mask(
                        predicted[:, :, 0], self.labels, self.config),
                    caption=f'Combined output prediction {i}'
                )
            )
        return mask_list

    def _in3_out4(self, pred_masks):
        val_masks = [
            tf.image.convert_image_dtype(self.val_masks[0], tf.uint8),
            tf.image.convert_image_dtype(self.val_masks[1], tf.uint8),
            tf.image.convert_image_dtype(self.val_masks[2], tf.uint8),
            tf.image.convert_image_dtype(self.val_masks[3], tf.uint8),

        ]
        val_images = [
            tf.image.convert_image_dtype(self.val_images[0], tf.uint8),
            tf.image.convert_image_dtype(self.val_images[1], tf.uint8),
            tf.image.convert_image_dtype(self.val_images[2], tf.uint8)
        ]
        pred_masks = [
            tf.image.convert_image_dtype(pred_masks[0], tf.uint8),
            tf.image.convert_image_dtype(pred_masks[1], tf.uint8),
            tf.image.convert_image_dtype(pred_masks[2], tf.uint8),
            tf.image.convert_image_dtype(pred_masks[3], tf.uint8),
        ]

        mask_list = list()
        for i in range(len(self.val_images)):
            for scale in range(3):
                mask_list.append(wandb_mask(val_images[scale][i].numpy()[
                                 :, :, :3], caption=f'Image {i} - scale {scale + 1}'))
                for idx_c, c in enumerate(self.config['classes']):
                    mask_list.append(
                        wandb_mask(
                            create_prediction_img(val_masks[scale][i].numpy()[
                                                  :, :, idx_c], pred_masks[scale][i].numpy()[:, :, idx_c]),
                            caption=f'Image {i} - {c}'
                        )
                    )

                true, predicted = combined_output(
                    val_masks[3][i].numpy(), pred_masks[3][i].numpy())
                mask_list.append(
                    wandb_mask(
                        create_RGB_mask(
                            true[:, :, 0], self.labels, self.config),
                        caption=f'Combined output true {i}'
                    )
                )

                mask_list.append(
                    wandb_mask(
                        create_RGB_mask(
                            predicted[:, :, 0], self.labels, self.config),
                        caption=f'Combined output prediction {i}'
                    )
                )

        return mask_list
