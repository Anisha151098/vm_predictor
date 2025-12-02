"""
GRU-based Models for VM Usage Prediction
Implements: Simple GRU, BiGRU, GRU+CNN, CNN-GRU-Attention
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import (
    Input, GRU, Bidirectional, Dense, Dropout,
    Conv1D, MaxPooling1D, Flatten, Concatenate,
    LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D
)


def build_simple_gru(
    input_shape: tuple,
    gru_units: int = 128,
    dense_units: int = 64,
    dropout_rate: float = 0.2
) -> Model:
    """
    Build Simple GRU model.

    Args:
        input_shape: Shape of input (sequence_length, n_features)
        gru_units: Number of GRU units
        dense_units: Number of dense layer units
        dropout_rate: Dropout rate

    Returns:
        Compiled Keras model
    """
    inputs = Input(shape=input_shape, name='input')

    # GRU layers
    x = GRU(gru_units, return_sequences=True, name='gru_1')(inputs)
    x = Dropout(dropout_rate)(x)
    x = GRU(gru_units // 2, return_sequences=False, name='gru_2')(x)
    x = Dropout(dropout_rate)(x)

    # Dense layers
    x = Dense(dense_units, activation='relu', name='dense_1')(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(1, activation='linear', name='output')(x)

    model = Model(inputs=inputs, outputs=outputs, name='Simple_GRU')

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae', 'mse']
    )

    return model


def build_bigru(
    input_shape: tuple,
    gru_units: int = 128,
    dense_units: int = 64,
    dropout_rate: float = 0.2
) -> Model:
    """
    Build Bidirectional GRU model.

    Args:
        input_shape: Shape of input (sequence_length, n_features)
        gru_units: Number of GRU units
        dense_units: Number of dense layer units
        dropout_rate: Dropout rate

    Returns:
        Compiled Keras model
    """
    inputs = Input(shape=input_shape, name='input')

    # Bidirectional GRU layers
    x = Bidirectional(GRU(gru_units, return_sequences=True), name='bigru_1')(inputs)
    x = Dropout(dropout_rate)(x)
    x = Bidirectional(GRU(gru_units // 2, return_sequences=False), name='bigru_2')(x)
    x = Dropout(dropout_rate)(x)

    # Dense layers
    x = Dense(dense_units, activation='relu', name='dense_1')(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(1, activation='linear', name='output')(x)

    model = Model(inputs=inputs, outputs=outputs, name='BiGRU')

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae', 'mse']
    )

    return model


def build_gru_cnn(
    input_shape: tuple,
    gru_units: int = 128,
    cnn_filters: int = 64,
    kernel_size: int = 3,
    dense_units: int = 64,
    dropout_rate: float = 0.2
) -> Model:
    """
    Build GRU+CNN hybrid model.
    GRU extracts temporal features, CNN captures local patterns.

    Args:
        input_shape: Shape of input (sequence_length, n_features)
        gru_units: Number of GRU units
        cnn_filters: Number of CNN filters
        kernel_size: CNN kernel size
        dense_units: Number of dense layer units
        dropout_rate: Dropout rate

    Returns:
        Compiled Keras model
    """
    inputs = Input(shape=input_shape, name='input')

    # GRU branch
    gru_out = GRU(gru_units, return_sequences=True, name='gru_1')(inputs)
    gru_out = Dropout(dropout_rate)(gru_out)
    gru_out = GRU(gru_units // 2, return_sequences=False, name='gru_2')(gru_out)

    # CNN branch
    cnn_out = Conv1D(cnn_filters, kernel_size, padding='same', activation='relu', name='conv_1')(inputs)
    cnn_out = MaxPooling1D(pool_size=2, name='pool_1')(cnn_out)
    cnn_out = Conv1D(cnn_filters // 2, kernel_size, padding='same', activation='relu', name='conv_2')(cnn_out)
    cnn_out = GlobalAveragePooling1D(name='global_pool')(cnn_out)

    # Concatenate GRU and CNN outputs
    x = Concatenate(name='concat')([gru_out, cnn_out])
    x = Dropout(dropout_rate)(x)

    # Dense layers
    x = Dense(dense_units, activation='relu', name='dense_1')(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(1, activation='linear', name='output')(x)

    model = Model(inputs=inputs, outputs=outputs, name='GRU_CNN')

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae', 'mse']
    )

    return model


class AttentionLayer(layers.Layer):
    """Custom attention layer for sequence data."""

    def __init__(self, units, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.W = self.add_weight(
            name='attention_weight',
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b = self.add_weight(
            name='attention_bias',
            shape=(self.units,),
            initializer='zeros',
            trainable=True
        )
        self.u = self.add_weight(
            name='attention_context',
            shape=(self.units,),
            initializer='glorot_uniform',
            trainable=True
        )
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        # x shape: (batch_size, time_steps, features)
        uit = tf.nn.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
        ait = tf.tensordot(uit, self.u, axes=1)
        a = tf.nn.softmax(ait, axis=1)
        weighted_input = x * tf.expand_dims(a, -1)
        output = tf.reduce_sum(weighted_input, axis=1)
        return output

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config


def build_cnn_gru_attention(
    input_shape: tuple,
    cnn_filters: int = 64,
    kernel_size: int = 3,
    gru_units: int = 128,
    attention_units: int = 64,
    dense_units: int = 64,
    dropout_rate: float = 0.2
) -> Model:
    """
    Build CNN-GRU-Attention model.
    CNN extracts local features, GRU captures temporal dependencies,
    Attention mechanism focuses on important time steps.

    Args:
        input_shape: Shape of input (sequence_length, n_features)
        cnn_filters: Number of CNN filters
        kernel_size: CNN kernel size
        gru_units: Number of GRU units
        attention_units: Number of attention units
        dense_units: Number of dense layer units
        dropout_rate: Dropout rate

    Returns:
        Compiled Keras model
    """
    inputs = Input(shape=input_shape, name='input')

    # CNN layers for feature extraction
    x = Conv1D(cnn_filters, kernel_size, padding='same', activation='relu', name='conv_1')(inputs)
    x = MaxPooling1D(pool_size=2, name='pool_1')(x)
    x = Dropout(dropout_rate)(x)

    x = Conv1D(cnn_filters // 2, kernel_size, padding='same', activation='relu', name='conv_2')(x)
    x = Dropout(dropout_rate)(x)

    # GRU layers for temporal modeling
    x = GRU(gru_units, return_sequences=True, name='gru_1')(x)
    x = Dropout(dropout_rate)(x)
    x = GRU(gru_units // 2, return_sequences=True, name='gru_2')(x)
    x = Dropout(dropout_rate)(x)

    # Attention mechanism
    x = AttentionLayer(attention_units, name='attention')(x)

    # Dense layers
    x = Dense(dense_units, activation='relu', name='dense_1')(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(1, activation='linear', name='output')(x)

    model = Model(inputs=inputs, outputs=outputs, name='CNN_GRU_Attention')

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae', 'mse']
    )

    return model


def get_model(model_name: str, input_shape: tuple, **kwargs) -> Model:
    """
    Factory function to get model by name.

    Args:
        model_name: Name of model ('simple_gru', 'bigru', 'gru_cnn', 'cnn_gru_attention')
        input_shape: Shape of input
        **kwargs: Additional model parameters

    Returns:
        Compiled Keras model
    """
    models = {
        'simple_gru': build_simple_gru,
        'bigru': build_bigru,
        'gru_cnn': build_gru_cnn,
        'cnn_gru_attention': build_cnn_gru_attention
    }

    if model_name.lower() not in models:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(models.keys())}")

    return models[model_name.lower()](input_shape, **kwargs)


if __name__ == "__main__":
    # Test model building
    input_shape = (50, 4)  # 50 time steps, 4 features

    print("Building models...\n")

    models = ['simple_gru', 'bigru', 'gru_cnn', 'cnn_gru_attention']

    for model_name in models:
        model = get_model(model_name, input_shape)
        print(f"✓ {model.name}")
        print(f"  Parameters: {model.count_params():,}")
        print(f"  Input shape: {model.input_shape}")
        print(f"  Output shape: {model.output_shape}\n")

    print("✓ All models built successfully!")
