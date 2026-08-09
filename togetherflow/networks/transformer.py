import bayesflow as bf


@bf.utils.serialization.serializable("custom")
class TransformerSummaryNet(bf.networks.TimeSeriesTransformer):
    """Time-series transformer summary network.

    Sized to the Base configuration. `time_axis` is left unset deliberately: the
    simulator emits no explicit time column, so the sequence carries only ordered
    observations and Time2Vec falls back to implicit positions.
    """

    def __init__(self, summary_dim: int = 16, **kwargs):
        kwargs.setdefault("embed_dims", (64, 64, 64))
        kwargs.setdefault("num_heads", (4, 4, 4))
        kwargs.setdefault("time_embed_dim", 8)
        kwargs.setdefault("dropout", 0.1)
        super().__init__(summary_dim=summary_dim, **kwargs)
