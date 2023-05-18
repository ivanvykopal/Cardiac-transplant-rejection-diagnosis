import tensorflow as tf


def get_optimizer(lr, params):
    name = params['name']
    params = {k: v for k, v in params.items() if k != 'name'}

    if name.lower() == 'adam':
        return tf.keras.optimizers.Adam(learning_rate=lr, **params)
    elif name.lower() == 'rmsprop':
        return tf.keras.optimizers.RMSprop(learning_rate=lr, **params)
    elif name.lower() == 'sgd':
        return tf.keras.optimizers.SGD(learning_rate=lr, **params)
    elif name.lower() == 'adagrad':
        return tf.keras.optimizers.Adagrad(learning_rate=lr, **params)
    elif name.lower() == 'adadelta':
        return tf.keras.optimizers.Adadelta(learning_rate=lr, **params)
