import keras
import bayesflow as bf
from keras import layers, ops


@bf.utils.serialization.serializable("custom")
class SummaryNet(bf.networks.SummaryNetwork):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.base = bf.networks.Sequential([
            keras.layers.Conv1D(filters=32, kernel_size=3, strides=3, activation="swish"),
            keras.layers.GroupNormalization(groups=8),
            keras.layers.Conv1D(filters=32, kernel_size=2, strides=2, activation="swish"),
            keras.layers.GroupNormalization(groups=8),
            keras.layers.Bidirectional(keras.layers.LSTM(256, dropout=0.2)),
            keras.layers.Dense(16)
        ])

        # self.base = bf.networks.DeepSet()

    def build(self, input_shape):
        self.base.build(input_shape)
        super().build(input_shape)  # marks self.built = True

    def call(self, x, training: bool = False):
        return self.base(x, training=training)
