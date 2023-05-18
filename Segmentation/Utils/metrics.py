import tensorflow as tf

from Utils.losses import diceCoef
from Utils.losses import diceCoefClass


def get_metrics(metrics):
    result = []
    for metric in metrics:
        metric_name = metric['name']
        metric_attr = {k: v for k, v in metric.items() if k != 'name'}

        if metric_name == 'AUC':
            result.append(tf.keras.metrics.AUC(**metric_attr))
            continue
        if metric_name == 'Accuracy':
            result.append(tf.keras.metrics.Accuracy(**metric_attr))
            continue
        if metric_name == 'BinaryAccuracy':
            result.append(tf.keras.metrics.BinaryAccuracy(**metric_attr))
            continue
        if metric_name == 'BinaryCrossentropy':
            result.append(tf.keras.metrics.BinaryCrossentropy(**metric_attr))
            continue
        if metric_name == 'BinaryIoU':
            result.append(tf.keras.metrics.BinaryIoU(**metric_attr))
            continue
        if metric_name == 'CategoricalAccuracy':
            result.append(tf.keras.metrics.CategoricalAccuracy(**metric_attr))
            continue
        if metric_name == 'CategoricalCrossentropy':
            result.append(tf.keras.metrics.CategoricalCrossentropy(**metric_attr))
            continue
        if metric_name == 'CategoricalHinge':
            result.append(tf.keras.metrics.CategoricalHinge(**metric_attr))
            continue
        if metric_name == 'CosineSimilarity':
            result.append(tf.keras.metrics.CosineSimilarity(**metric_attr))
            continue
        if metric_name == 'FalseNegatives':
            result.append(tf.keras.metrics.FalseNegatives(**metric_attr))
            continue
        if metric_name == 'FalsePositives':
            result.append(tf.keras.metrics.FalsePositives(**metric_attr))
            continue
        if metric_name == 'Hinge':
            result.append(tf.keras.metrics.Hinge(**metric_attr))
            continue
        if metric_name == 'KLDivergence':
            result.append(tf.keras.metrics.KLDivergence(**metric_attr))
            continue
        if metric_name == 'LogCoshError':
            result.append(tf.keras.metrics.LogCoshError(**metric_attr))
            continue
        if metric_name == 'Mean':
            result.append(tf.keras.metrics.Mean(**metric_attr))
            continue
        if metric_name == 'MeanAbsoluteError':
            result.append(tf.keras.metrics.MeanAbsoluteError(**metric_attr))
            continue
        if metric_name == 'MeanAbsolutePercentageError':
            result.append(tf.keras.metrics.MeanAbsolutePercentageError(**metric_attr))
            continue
        if metric_name == 'MeanIoU':
            result.append(tf.keras.metrics.MeanIoU(**metric_attr))
            continue
        if metric_name == 'MeanMetricWrapper':
            result.append(tf.keras.metrics.MeanMetricWrapper(**metric_attr))
            continue
        if metric_name == 'MeanRelativeError':
            result.append(tf.keras.metrics.MeanRelativeError(**metric_attr))
            continue
        if metric_name == 'MeanSquaredError':
            result.append(tf.keras.metrics.MeanSquaredError(**metric_attr))
            continue
        if metric_name == 'MeanSquaredLogarithmicError':
            result.append(tf.keras.metrics.MeanSquaredLogarithmicError(**metric_attr))
            continue
        if metric_name == 'MeanTensor':
            result.append(tf.keras.metrics.MeanTensor(**metric_attr))
            continue
        if metric_name == 'OneHotIoU':
            result.append(tf.keras.metrics.OneHotIoU(**metric_attr))
            continue
        if metric_name == 'OneHotMeanIoU':
            result.append(tf.keras.metrics.OneHotMeanIoU(**metric_attr))
            continue
        if metric_name == 'Poisson':
            result.append(tf.keras.metrics.Poisson(**metric_attr))
            continue
        if metric_name == 'Precision':
            result.append(tf.keras.metrics.Precision(**metric_attr))
            continue
        if metric_name == 'PrecisionAtRecall':
            result.append(tf.keras.metrics.PrecisionAtRecall(**metric_attr))
            continue
        if metric_name == 'Recall':
            result.append(tf.keras.metrics.Recall(**metric_attr))
            continue
        if metric_name == 'RecallAtPrecision':
            result.append(tf.keras.metrics.RecallAtPrecision(**metric_attr))
            continue
        if metric_name == 'RootMeanSquaredError':
            result.append(tf.keras.metrics.RootMeanSquaredError(**metric_attr))
            continue
        if metric_name == 'SensitivityAtSpecificity':
            result.append(tf.keras.metrics.SensitivityAtSpecificity(**metric_attr))
            continue
        if metric_name == 'SparseCategoricalAccuracy':
            result.append(tf.keras.metrics.SparseCategoricalAccuracy(**metric_attr))
            continue
        if metric_name == 'SparseCategoricalCrossentropy':
            result.append(tf.keras.metrics.SparseCategoricalCrossentropy(**metric_attr))
            continue
        if metric_name == 'SparseTopKCategoricalAccuracy':
            result.append(tf.keras.metrics.SparseTopKCategoricalAccuracy(**metric_attr))
            continue
        if metric_name == 'SpecificityAtSensitivity':
            result.append(tf.keras.metrics.SpecificityAtSensitivity(**metric_attr))
            continue
        if metric_name == 'SquaredHinge':
            result.append(tf.keras.metrics.SquaredHinge(**metric_attr))
            continue
        if metric_name == 'Sum':
            result.append(tf.keras.metrics.Sum(**metric_attr))
            continue
        if metric_name == 'TopKCategoricalAccuracy':
            result.append(tf.keras.metrics.TopKCategoricalAccuracy(**metric_attr))
            continue
        if metric_name == 'TrueNegatives':
            result.append(tf.keras.metrics.TrueNegatives(**metric_attr))
            continue
        if metric_name == 'TruePositives':
            result.append(tf.keras.metrics.TruePositives(**metric_attr))
            continue
        if metric_name == 'diceCoef':
            result.append(diceCoef)
            continue
        if metric_name == 'diceCoefClass':
            result.append(diceCoefClass)
            continue

    return result
