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
            keras.layers.LSTM(256, dropout=0.2),
            keras.layers.Dense(16)
        ])

    def build(self, input_shape):
        self.base.build(input_shape)
        super().build(input_shape)  # marks self.built = True

    def call(self, x, training: bool = False):
        return self.base(x, training=training)


class AgentTemporalEncoder(layers.Layer):
    """Shared per-agent time-series encoder: (B, T, F) -> (B, D)."""

    def __init__(self, d_model=128, conv_k=5, gru_units=128, dt=1.0, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.dt = dt
        self.conv1 = layers.Conv1D(d_model, conv_k, padding="same")
        self.conv2 = layers.Conv1D(d_model, conv_k, padding="same")
        self.norm1 = layers.LayerNormalization()
        self.gru = layers.GRU(gru_units, return_sequences=True)
        self.norm2 = layers.LayerNormalization()
        self.dropout = layers.Dropout(dropout)
        self.proj = layers.Dense(d_model, activation="gelu")

    def _preprocess(self, x):
        # x: (B, T, F) with F=[x, y, theta, k_neighbors]
        pos = x[..., :2]  # (B,T,2)
        th = x[..., 2:3]  # (B,T,1)
        kn = x[..., 3:4]  # (B,T,1)

        dpos = ops.pad(pos[:, 1:, :] - pos[:, :-1, :], [[0, 0], [1, 0], [0, 0]])
        speed = ops.sqrt(ops.sum(dpos ** 2, axis=-1, keepdims=True)) / self.dt

        th_sin = ops.sin(th)
        th_cos = ops.cos(th)
        dth = ops.pad(th[:, 1:, :] - th[:, :-1, :], [[0, 0], [1, 0], [0, 0]])
        dth = ops.atan2(ops.sin(dth), ops.cos(dth))
        dth_sin = ops.sin(dth)
        dth_cos = ops.cos(dth)

        feats = ops.concatenate([dpos, speed, th_sin, th_cos, dth_sin, dth_cos, kn], axis=-1)
        return feats  # (B,T,8)

    def call(self, x, training=False):
        z = self._preprocess(x)  # (B,T,8)
        z = self.conv1(z);
        z = ops.gelu(z);
        z = self.norm1(z)
        z = self.conv2(z);
        z = ops.gelu(z)
        z = self.gru(z, training=training)  # (B,T,gru_units)
        z = self.norm2(z)
        z_mean = ops.mean(z, axis=1)
        z_max = ops.max(z, axis=1)
        z = ops.concatenate([z_mean, z_max], axis=-1)  # (B, 2*gru_units)
        return self.proj(z)  # (B, D)

class TogetherNet(keras.Model):
    """
    Base summary network to plug into SummaryNet.

    Accepts:
      - (B, T, N, F)  with F=4: [x, y, theta, k_neighbors]
      - (B, T, N*F)   flattened across agents

    Outputs:
      - (B, 2): [r_hat, v_hat] (softplus, positive)
    """
    def __init__(self,
                 d_model=128,
                 conv_k=5,
                 gru_units=128,
                 dt=1.0,
                 n_heads=4,
                 dropout=0.1,
                 hidden=256,
                 r_scale=1.0,
                 v_scale=1.0,
                 num_features=4):
        super().__init__()
        self.enc = AgentTemporalEncoder(d_model, conv_k, gru_units, dt, dropout)
        self.norm_agents = layers.LayerNormalization()
        self.mha = layers.MultiHeadAttention(num_heads=n_heads, key_dim=d_model)
        self.query = self.add_weight(
            name="agent_query", shape=(1, 1, d_model),
            initializer="zeros", trainable=True
        )
        self.dropout = layers.Dropout(dropout)
        self.head = keras.Sequential([
            layers.Dense(hidden, activation="gelu"),
            layers.Dense(hidden//2, activation="gelu"),
            layers.Dense(2)  # raw outputs
        ])
        self.r_scale = r_scale
        self.v_scale = v_scale
        self.num_features = num_features  # F (default 4)

    def call(self, x, training=False):
        # Auto-unflatten (B, T, N*F) -> (B, T, N, F)
        if x.ndim == 3:
            B, T, NF = x.shape
            F = self.num_features
            if NF % F != 0:
                raise ValueError(f"Last dim {NF} is not divisible by num_features={F}.")
            N = NF // F
            x = ops.reshape(x, (B, T, N, F))

        # x now (B, T, N, F)
        B, T, N, F = x.shape
        x_ = ops.reshape(x, (B * N, T, F))                 # per-agent batch
        e = self.enc(x_, training=training)                 # (B*N, D)
        D = e.shape[-1]
        e = ops.reshape(e, (B, N, D))                       # (B, N, D)
        e = self.norm_agents(e)

        # permutation-invariant pooling across agents
        q = ops.tile(self.query, (B, 1, 1))                 # (B,1,D)
        attn = self.mha(query=q, value=e, key=e, training=training)  # (B,1,D)
        attn = ops.squeeze(attn, axis=1)                    # (B,D)
        mean_pool = ops.mean(e, axis=1)                     # (B,D)
        max_pool  = ops.max(e, axis=1)                      # (B,D)
        g = ops.concatenate([attn, mean_pool, max_pool], axis=-1)    # (B,3D)
        g = self.dropout(g, training=training)

        out_raw = self.head(g)                               # (B,2)
        out_pos = ops.softplus(out_raw)
        r_hat = out_pos[:, :1] * self.r_scale
        v_hat = out_pos[:, 1:2] * self.v_scale
        return ops.concatenate([r_hat, v_hat], axis=-1) # (B,2)

    def compute_output_shape(self, input_shape):
        # Supports (B, T, N, F) and (B, T, N*F); output is (B, 2)
        if isinstance(input_shape, (list, tuple)) and len(input_shape) >= 1:
            B = input_shape[0]
        else:
            B = None
        return (B, 2)
