import keras
import bayesflow as bf


class GRU(bf.networks.SummaryNetwork):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.gru = keras.layers.GRU(64, dropout=0.1)
        self.summary_stats = keras.layers.Dense(8)

    def call(self, time_series, **kwargs):
        """Compresses time_series of shape (batch_size, T, 1) into summaries of shape (batch_size, 8)."""

        summary = self.gru(time_series, training=kwargs.get("stage") == "training")
        summary = self.summary_stats(summary)
        return summary

class SummaryNet(keras.Model):

    def __init__(self, base):
        super().__init__()
        self.base = base

    def build(self, input_shape):
        self.base.build(input_shape)
        super().build(input_shape)  # marks self.built = True

    def call(self, x, training: bool=False):
        return self.base(x, training=training)

    def compute_output_shape(self, input_shape):
        return self.base.compute_output_shape(input_shape)

    def compute_metrics(self, inputs, *, stage: str | None = None, sample_weight=None, **kwargs):
        training_flag = (stage == "training")
        outputs = self(inputs, training=training_flag)
        # Zero dummy loss so the API is satisfied
        loss = keras.ops.zeros(())
        return {"loss": loss, "outputs": outputs}
