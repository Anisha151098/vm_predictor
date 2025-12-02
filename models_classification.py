"""
GRU-based Models for VM CPU Classification
Implements: Simple GRU, BiGRU, GRU+CNN, CNN-GRU-Attention, GRU+LSTM
Classification task: High CPU (1) vs Low CPU (0)
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import (
    Input, GRU, LSTM, Bidirectional, Dense, Dropout,
    Conv1D, MaxPooling1D, Flatten, Concatenate,
    LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D
)


def build_simple_gru_classifier(
    input_shape: tuple,
    gru_units: int = 128,
    dense_units: int = 64,
    dropout_rate: float = 0.3
) -> Model:
    """Simple GRU for binary classification."""
    inputs = Input(shape=input_shape, name='input')

    # GRU layers
    x = GRU(gru_units, return_sequences=True, name='gru_1')(inputs)
    x = Dropout(dropout_rate)(x)
    x = GRU(gru_units // 2, return_sequences=False, name='gru_2')(x)
    x = Dropout(dropout_rate)(x)

    # Dense layers
    x = Dense(dense_units, activation='relu', name='dense_1')(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(1, activation='sigmoid', name='output')(x)

    model = Model(inputs=inputs, outputs=outputs, name='Simple_GRU')

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc'),
                 keras.metrics.Precision(name='precision'),
                 keras.metrics.Recall(name='recall')]
    )

    return model


def build_bigru_classifier(
    input_shape: tuple,
    gru_units: int = 128,
    dense_units: int = 64,
    dropout_rate: float = 0.3
) -> Model:
    """Bidirectional GRU for binary classification."""
    inputs = Input(shape=input_shape, name='input')

    # Bidirectional GRU layers
    x = Bidirectional(GRU(gru_units, return_sequences=True), name='bigru_1')(inputs)
    x = Dropout(dropout_rate)(x)
    x = Bidirectional(GRU(gru_units // 2, return_sequences=False), name='bigru_2')(x)
    x = Dropout(dropout_rate)(x)

    # Dense layers
    x = Dense(dense_units, activation='relu', name='dense_1')(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(1, activation='sigmoid', name='output')(x)

    model = Model(inputs=inputs, outputs=outputs, name='BiGRU')

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc'),
                 keras.metrics.Precision(name='precision'),
                 keras.metrics.Recall(name='recall')]
    )

    return model


def build_gru_cnn_classifier(
    input_shape: tuple,
    gru_units: int = 128,
    cnn_filters: int = 64,
    kernel_size: int = 3,
    dense_units: int = 64,
    dropout_rate: float = 0.3
) -> Model:
    """GRU+CNN hybrid for binary classification."""
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

    # Concatenate
    x = Concatenate(name='concat')([gru_out, cnn_out])
    x = Dropout(dropout_rate)(x)

    # Dense layers
    x = Dense(dense_units, activation='relu', name='dense_1')(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(1, activation='sigmoid', name='output')(x)

    model = Model(inputs=inputs, outputs=outputs, name='GRU_CNN')

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc'),
                 keras.metrics.Precision(name='precision'),
                 keras.metrics.Recall(name='recall')]
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


def build_cnn_gru_attention_classifier(
    input_shape: tuple,
    cnn_filters: int = 64,
    kernel_size: int = 3,
    gru_units: int = 128,
    attention_units: int = 64,
    dense_units: int = 64,
    dropout_rate: float = 0.3
) -> Model:
    """CNN-GRU-Attention for binary classification."""
    inputs = Input(shape=input_shape, name='input')

    # CNN layers
    x = Conv1D(cnn_filters, kernel_size, padding='same', activation='relu', name='conv_1')(inputs)
    x = MaxPooling1D(pool_size=2, name='pool_1')(x)
    x = Dropout(dropout_rate)(x)

    x = Conv1D(cnn_filters // 2, kernel_size, padding='same', activation='relu', name='conv_2')(x)
    x = Dropout(dropout_rate)(x)

    # GRU layers
    x = GRU(gru_units, return_sequences=True, name='gru_1')(x)
    x = Dropout(dropout_rate)(x)
    x = GRU(gru_units // 2, return_sequences=True, name='gru_2')(x)
    x = Dropout(dropout_rate)(x)

    # Attention mechanism
    x = AttentionLayer(attention_units, name='attention')(x)

    # Dense layers
    x = Dense(dense_units, activation='relu', name='dense_1')(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(1, activation='sigmoid', name='output')(x)

    model = Model(inputs=inputs, outputs=outputs, name='CNN_GRU_Attention')

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc'),
                 keras.metrics.Precision(name='precision'),
                 keras.metrics.Recall(name='recall')]
    )

    return model


def build_gru_lstm_classifier(
    input_shape: tuple,
    gru_units: int = 128,
    lstm_units: int = 128,
    dense_units: int = 64,
    dropout_rate: float = 0.3
) -> Model:
    """GRU+LSTM hybrid for binary classification."""
    inputs = Input(shape=input_shape, name='input')

    # GRU layers
    x = GRU(gru_units, return_sequences=True, name='gru_1')(inputs)
    x = Dropout(dropout_rate)(x)

    # LSTM layers
    x = LSTM(lstm_units, return_sequences=True, name='lstm_1')(x)
    x = Dropout(dropout_rate)(x)
    x = LSTM(lstm_units // 2, return_sequences=False, name='lstm_2')(x)
    x = Dropout(dropout_rate)(x)

    # Dense layers
    x = Dense(dense_units, activation='relu', name='dense_1')(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(1, activation='sigmoid', name='output')(x)

    model = Model(inputs=inputs, outputs=outputs, name='GRU_LSTM')

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc'),
                 keras.metrics.Precision(name='precision'),
                 keras.metrics.Recall(name='recall')]
    )

    return model


def get_model(model_name: str, input_shape: tuple, **kwargs) -> Model:
    """Factory function to get model by name."""
    models = {
        'simple_gru': build_simple_gru_classifier,
        'bigru': build_bigru_classifier,
        'gru_cnn': build_gru_cnn_classifier,
        'cnn_gru_attention': build_cnn_gru_attention_classifier,
        'gru_lstm': build_gru_lstm_classifier
    }

    if model_name.lower() not in models:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(models.keys())}")

    return models[model_name.lower()](input_shape, **kwargs)
