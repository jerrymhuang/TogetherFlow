import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = models.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim)
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class PositionalEncoding(layers.Layer):
    def __init__(self, sequence_length, embed_dim):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self._get_positional_encoding(sequence_length, embed_dim)

    def _get_positional_encoding(self, length, depth):
        pos = np.arange(length)[:, np.newaxis]
        i = np.arange(depth)[np.newaxis, :]
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(depth))
        angle_rads = pos * angle_rates

        # Apply sin to even indices; cos to odd indices
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        return tf.cast(angle_rads[np.newaxis, ...], dtype=tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]


def create_space_time_transformer(time_steps, tokens, embed_dim, num_heads, ff_dim, num_layers):
    inputs = layers.Input(shape=(time_steps, tokens, embed_dim))

    # Spatial Transformer Block
    x = tf.reshape(inputs, (-1, tokens, embed_dim))  # (Batch * Time, Tokens, Embed_dim)
    x = PositionalEncoding(tokens, embed_dim)(x)
    for _ in range(num_layers):
        x = TransformerBlock(embed_dim, num_heads, ff_dim)(x)
    x = tf.reshape(x, (-1, time_steps, tokens, embed_dim))  # Back to (Batch, Time, Tokens, Embed_dim)

    # Temporal Transformer Block
    x = tf.transpose(x, perm=[0, 2, 1, 3])  # (Batch, Tokens, Time, Embed_dim)
    x = tf.reshape(x, (-1, time_steps, embed_dim))  # (Batch * Tokens, Time, Embed_dim)
    x = PositionalEncoding(time_steps, embed_dim)(x)
    for _ in range(num_layers):
        x = TransformerBlock(embed_dim, num_heads, ff_dim)(x)
    x = tf.reshape(x, (-1, tokens, time_steps, embed_dim))
    x = tf.transpose(x, perm=[0, 2, 1, 3])  # (Batch, Time, Tokens, Embed_dim)

    # Global Pooling and Output
    x = layers.GlobalAveragePooling2D()(x)  # Pool over both time and tokens
    outputs = layers.Dense(1, activation="sigmoid")(x)  # Example: Binary classification

    return models.Model(inputs=inputs, outputs=outputs)


def create_agent_space_time_transformer(time_steps, num_agents, feature_dim, embed_dim, num_heads, ff_dim, num_layers):
    inputs = layers.Input(shape=(time_steps, num_agents, feature_dim))

    # Linear projection to embedding dimension
    x = layers.Dense(embed_dim)(inputs)

    # Temporal Positional Encoding
    time_encoding = layers.Embedding(input_dim=time_steps, output_dim=embed_dim)
    time_indices = tf.range(start=0, limit=time_steps, delta=1)
    x += time_encoding(time_indices)[None, :, None, :]

    # Spatial Transformer (Agent Relationships per Time Step)
    x_spatial = tf.reshape(x, (-1, num_agents, embed_dim))  # (Batch * Time, Agents, Embed_dim)
    for _ in range(num_layers):
        x_spatial = TransformerBlock(embed_dim, num_heads, ff_dim)(x_spatial)
    x_spatial = tf.reshape(x_spatial, (-1, time_steps, num_agents, embed_dim))  # Back to (Batch, Time, Agents, Embed_dim)

    # Temporal Transformer (Agent Evolution over Time)
    x_temporal = tf.transpose(x_spatial, perm=[0, 2, 1, 3])  # (Batch, Agents, Time, Embed_dim)
    x_temporal = tf.reshape(x_temporal, (-1, time_steps, embed_dim))  # (Batch * Agents, Time, Embed_dim)
    for _ in range(num_layers):
        x_temporal = TransformerBlock(embed_dim, num_heads, ff_dim)(x_temporal)
    x_temporal = tf.reshape(x_temporal, (-1, num_agents, time_steps, embed_dim))
    x_temporal = tf.transpose(x_temporal, perm=[0, 2, 1, 3])  # (Batch, Time, Agents, Embed_dim)

    # Global Pooling
    x = layers.GlobalAveragePooling2D()(x_temporal)
    outputs = layers.Dense(1, activation="sigmoid")(x)  # Example: Binary classification

    return models.Model(inputs=inputs, outputs=outputs)


class SpaceTimeTransformer(tf.keras.Model):
    def __init__(self):
        super(SpaceTimeTransformer, self).__init__()

    def call(self, inputs):
        raise NotImplementedError


class AgentSpaceTimeTransformer(tf.keras.Model):
    def __init__(self):
        super(AgentSpaceTimeTransformer, self).__init__()

    def call(self, inputs):
        raise NotImplementedError
