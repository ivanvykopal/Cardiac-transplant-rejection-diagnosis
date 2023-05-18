import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np
import keras.backend as K


def tversky(y_true, y_pred, smooth=1e-5):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)

    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1 - y_pred_pos))
    false_pos = K.sum((1 - y_true_pos) * y_pred_pos)

    if true_pos + false_pos < false_neg:
        return 1.

    alpha = 0.7
    tp_weight = 3
    return (tp_weight * true_pos + smooth) / (tp_weight * true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)


def focal_tversky(y_true, y_pred):
    pt_1 = tversky(y_true, y_pred)
    gamma = 0.75
    return K.pow((1 - pt_1), gamma)


def binary_crossentropy(y_true, y_pred):
    return K.mean(K.binary_crossentropy(y_true, y_pred))


def diceCoef(y_true, y_pred, smooth=1):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)

    intersection = K.sum(y_true * y_pred)
    return (2 * intersection) / (K.sum(y_true) + K.sum(y_pred))


def diceCoefLoss(y_true, y_pred):
    return 1 - diceCoef(y_true, y_pred)


def diceCoefClass(y_true, y_pred, smooth=1):
    y_true = K.flatten(y_true[Ellipsis, 1:])
    y_pred = K.flatten(y_pred[..., 1:])

    intersection = K.sum(y_true * y_pred, axis=-1)
    denom = K.sum(y_true + y_pred, axis=-1)
    return K.mean((2. * intersection) / (denom + smooth))


def diceCoefClassLoss(y_true, y_pred):
    return 1 - diceCoefClass(y_true, y_pred)


def tversky_loss(y_true, y_pred):
    return 1.0 - tversky(y_true, y_pred)


def IoU(y_true, y_pred, smooth=1e-6):
    intersection = K.sum(K.sum(K.abs(y_true * y_pred), axis=-1))
    sum_ = K.sum(K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1))
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return jac


def IoULoss(y_true, y_pred, smooth=1e-6):
    intersection = K.sum(K.sum(K.abs(y_true * y_pred), axis=-1))
    sum_ = K.sum(K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1))
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth


def weighted_bincrossentropy(true, pred, weight_zero=0.25, weight_one=10):
    # calculate the binary cross entropy
    bin_crossentropy = K.binary_crossentropy(true, pred)

    # apply the weights
    weights = true * weight_one + (1. - true) * weight_zero
    weighted_bin_crossentropy = weights * bin_crossentropy

    return K.mean(weighted_bin_crossentropy)


def Combo_loss(targets, inputs, eps=1e-9, smooth=1e-6, alpha=0.5, ce_ration=0.5):
    targets = K.flatten(targets)
    inputs = K.flatten(inputs)

    intersection = K.sum(targets * inputs)
    dice = (2. * intersection + smooth) / (K.sum(targets) + K.sum(inputs) + smooth)
    inputs = K.clip(inputs, eps, 1.0 - eps)
    out = - (alpha * ((targets * K.log(inputs)) + ((1 - alpha) * (1.0 - targets) * K.log(1.0 - inputs))))
    weighted_ce = K.mean(out, axis=-1)
    combo = (ce_ration * weighted_ce) - ((1 - ce_ration) * dice)

    return combo


def FocalClassLoss(targets, inputs, alpha=0.8, gamma=2):
    targets = K.flatten(targets[Ellipsis, 1:])
    inputs = K.flatten(inputs[..., 1:])

    BCE = K.binary_crossentropy(targets, inputs)
    BCE_EXP = K.exp(-BCE)
    focal_loss = K.mean(alpha * K.pow((1 - BCE_EXP), gamma) * BCE)

    return focal_loss


def FocalLoss(targets, inputs, alpha=0.8, gamma=2):
    targets = K.flatten(targets)
    inputs = K.flatten(inputs)

    BCE = K.binary_crossentropy(targets, inputs)
    BCE_EXP = K.exp(-BCE)
    focal_loss = K.mean(alpha * K.pow((1 - BCE_EXP), gamma) * BCE)

    return focal_loss


def custom_metric(y_true, y_pred, smooth=1e-6, alpha=0.8):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)

    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1 - y_pred_pos))
    false_pos = K.sum((1 - y_true_pos) * y_pred_pos)
    return (2 * true_pos + smooth) / (2 * true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)


def weighted_categorical_crossentropy(weights):
    def wcce(y_true, y_pred):
        Kweights = K.constant(weights)
        y_true = K.cast(y_true, y_pred.dtype)
        return K.categorical_crossentropy(y_true, y_pred) * K.sum(y_true * Kweights, axis=-1)

    return wcce


def categorical_focal_loss(alpha, gamma=2.):
    alpha = np.array(alpha, dtype=np.float32)

    def categorical_focal_loss_fixed(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

        cross_entropy = -y_true * K.log(y_pred)

        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy
        return K.mean(K.sum(loss, axis=-1))

    return categorical_focal_loss_fixed


def soft_dice_loss(y_true, y_pred, epsilon=1e-6):
    axes = tuple(range(1, len(y_pred.shape)-1))
    numerator = 2. * np.sum(y_pred * y_true, axes)
    denominator = np.sum(np.square(y_pred) + np.square(y_true), axes)

    return 1 - np.mean((numerator + epsilon) / (denominator + epsilon)) # average over classes and batch


def DiceBCELoss(y_true, y_pred):
    dice_loss = diceCoefLoss(y_true, y_pred)
    bce = K.binary_crossentropy(y_true, y_pred)

    return bce + dice_loss


def DiceFocalLoss(y_true, y_pred):
    dice_loss = diceCoefLoss(y_true, y_pred)
    focal = FocalLoss(y_true, y_pred)

    return focal + dice_loss


def get_loss(loss_name, classes, weights=None):
    losses = {
        'diceCoefLoss': diceCoefLoss,
        'diceCoef': diceCoef,
        'BinaryCrossentropy': tf.keras.losses.BinaryCrossentropy(),
        'BinaryFocalCrossentropy': tf.keras.losses.BinaryFocalCrossentropy(),
        'CategoricalCrossentropy': tf.keras.losses.CategoricalCrossentropy(),
        'CategoricalHinge': tf.keras.losses.CategoricalHinge(),
        'CosineSimilarity': tf.keras.losses.CosineSimilarity(),
        'Hinge': tf.keras.losses.Hinge(),
        'Huber': tf.keras.losses.Huber(),
        'KLDivergence': tf.keras.losses.KLDivergence(),
        'LogCosh': tf.keras.losses.LogCosh(),
        'MeanAbsoluteError': tf.keras.losses.MeanAbsoluteError(),
        'MeanAbsolutePercentageError': tf.keras.losses.MeanAbsolutePercentageError(),
        'MeanSquaredError': tf.keras.losses.MeanSquaredError(),
        'MeanSquaredLogarithmicError': tf.keras.losses.MeanSquaredLogarithmicError(),
        'Poisson': tf.keras.losses.Poisson(),
        'SparseCategoricalCrossentropy': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        'SquaredHinge': tf.keras.losses.SquaredHinge(),
        'TverskyLoss': tversky_loss,
        'IoULoss': IoULoss,
        'BinaryCrossentropyMean': binary_crossentropy,
        'WeightedBinCrossentropy': weighted_bincrossentropy,
        'ComboLoss': Combo_loss,
        'CategoricalFocalLoss': categorical_focal_loss(weights),
        'WeightedCrossentropy': weighted_categorical_crossentropy(weights),
        'SoftDice': soft_dice_loss,
        'diceCoefClassLoss': diceCoefClassLoss,
        'FocalLoss': FocalLoss,
        'DiceBCELoss': DiceBCELoss,
        'DiceFocalLoss': DiceFocalLoss
    }

    if losses[loss_name] is None:
        return diceCoefLoss
    return losses[loss_name]
