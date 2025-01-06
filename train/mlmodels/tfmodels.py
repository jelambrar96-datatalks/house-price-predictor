from .basicmodel import BasicModel

import tensorflow as tf


class TensorFlowModel(BasicModel):

    def __init__(self, *args, **kwargs):
        self.shape = kwargs.get('shape', None)
        self.dropout = kwargs.get('dropout', None)
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation='elu', input_shape=(self.shape[1],)),
            tf.keras.layers.Dropout(self.dropout),
            tf.keras.layers.Dense(128, activation='elu'),
            tf.keras.layers.Dropout(self.dropout),
            tf.keras.layers.Dense(64, activation='elu'),
            tf.keras.layers.Dropout(self.dropout),
            tf.keras.layers.Dense(32, activation='elu'),
            tf.keras.layers.Dropout(self.dropout),
            tf.keras.layers.Dense(1)       # Capa de salida
        ])
