from tensorflow.keras.layers import RandomRotation, RandomFlip, RandomZoom, RandomContrast
import albumentations as A


# Types of augmentation:
#   - normal: {"type": "normal"}
#   - rotation: {"type": "rotation", "factor": 2}
#   - flip: {"type": "flip", "mode": "horizontal"}
#   - zoom: {"type": "zoom", "height_factor": 2, "width_factor": 2}
#   - contrast: {"type": "contrast, "factor": 2}
#   - transpose: {"type": "transpose"}
def augmentate(img, mask, augmentation):
    aug = None
    if augmentation['type'] == 'rotation':
        img = RandomRotation(factor=augmentation['factor'], interpolation='nearest', seed=42)(img)
        mask = RandomRotation(factor=augmentation['factor'], interpolation='nearest', seed=42)(mask)
    elif augmentation['type'] == 'flip':
        if augmentation['mode'] == 'horizontal':
            aug = A.HorizontalFlip(p=1)
        elif augmentation['mode'] == 'vertical':
            aug = A.VerticalFlip(p=1)
        elif augmentation['mode'] == 'horizontal_vertical':
            aug = A.Compose([
                A.HorizontalFlip(p=1),
                A.VerticalFlip(p=1)
            ])
    elif augmentation['type'] == 'zoom':
        img = RandomZoom(height_factor=augmentation['height_factor'],
                         width_factor=augmentation['width_factor'], interpolation='nearest', seed=42)(img)
        mask = RandomZoom(height_factor=augmentation['height_factor'],
                          width_factor=augmentation['width_factor'], interpolation='nearest', seed=42)(mask)
    elif augmentation['type'] == 'contrast':
        img = RandomContrast(factor=augmentation['factor'], seed=42)(img)
    elif augmentation['type'] == 'transpose':
        aug = A.Transpose(p=1)

    if aug:
        augmented = aug(image=img, mask=mask)
        img = augmented['image']
        mask = augmented['mask']
    return img, mask
