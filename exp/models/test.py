import numpy as np
import tensorflow as tf
import torch.nn as nn

from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2


class TorchTestModel(TorchModelV2, nn.Module):
    def __init__(self, *args, **kwargs):
        TorchModelV2.__init__(self, *args, **kwargs)
        nn.Module.__init__(self)

        self._shared = nn.Sequential(
            nn.Linear(np.product(self.obs_space.shape), 32),
            nn.PReLU(),
            nn.Linear(32, 64),
            nn.PReLU(),
            nn.Linear(64, 8),
        )

        self._logits = nn.Sequential(
            nn.Linear(8, self.num_outputs),
        )
        self._value = nn.Sequential(
            nn.Linear(8, 1),
        )

    def forward(self, input_dict, state, seq_lens):
        features = self._shared(input_dict["obs"])
        self._value_out = self._value(features)
        return self._logits(features), state

    def value_function(self):
        return self._value_out


class TensorflowTestModel(TFModelV2):
    def __init__(self, *args, **kwargs):
        TFModelV2.__init__(self, *args, **kwargs)

        inputs = tf.keras.layers.Input(shape=(np.product(self.obs_space.shape), ))

        hidden = tf.keras.layers.Dense(128, activation=tf.keras.activations.tanh, kernel_initializer=tf.keras.initializers.RandomNormal(stddev=1.0))(inputs)
        hidden = tf.keras.layers.Dense(32, kernel_initializer=tf.keras.initializers.RandomNormal(stddev=1.0))(hidden)
        logits = tf.keras.layers.Dense(self.num_outputs, kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01))(hidden)

        hidden = tf.keras.layers.Dense(128, activation=tf.keras.activations.tanh, kernel_initializer=tf.keras.initializers.RandomNormal(stddev=1.0))(inputs)
        hidden = tf.keras.layers.Dense(32, kernel_initializer=tf.keras.initializers.RandomNormal(stddev=1.0))(hidden)
        value = tf.keras.layers.Dense(1, kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01))(hidden)

        self._model = tf.keras.Model(inputs=inputs, outputs=[logits, value])
        self.register_variables(self._model.variables)

    def forward(self, input_dict, state, seq_lens):
        model_out, self._value_out = self._model(input_dict["obs_flat"])
        return model_out, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])
