import keras
import bayesflow as bf
from bayesflow.networks.helpers.sequential import Sequential


@bf.utils.serialization.serializable("custom")
class SummaryNet(bf.networks.SummaryNetwork):

    def __init__(self, summary_dim: int = 16, **kwargs):
        super().__init__(**kwargs)
        self.summary_dim = summary_dim

        self.base = Sequential([
            keras.layers.Conv1D(filters=32, kernel_size=3, strides=3, activation="swish"),
            keras.layers.GroupNormalization(groups=8),
            keras.layers.Conv1D(filters=32, kernel_size=2, strides=2, activation="swish"),
            keras.layers.GroupNormalization(groups=8),
            keras.layers.Bidirectional(keras.layers.LSTM(256, dropout=0.2)),
            keras.layers.Dense(summary_dim)
        ])

    def get_config(self):
        return super().get_config() | {"summary_dim": self.summary_dim}

    def build(self, input_shape):
        self.base.build(input_shape)
        super().build(input_shape)

    def call(self, x, training: bool = False):
        return self.base(x, training=training)
