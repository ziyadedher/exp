import numpy as np
import tensorflow as tf
from tensorflow.keras import activations, initializers, layers

from ray.rllib.models.tf.tf_modelv2 import TFModelV2


class ConfounderLatent(TFModelV2):
    def __init__(self, *args, **kwargs):
        TFModelV2.__init__(self, *args, **kwargs)

        inputs = layers.Input(shape=(np.product(self.obs_space.shape), ))

        # splits the observation from the confounders (last entry is confounder)
        obs, conf = layers.Lambda(lambda x: (x[:, :-1], x[:, -1]))(inputs)
        conf = tf.expand_dims(conf, 1)

        conf_lat = self._confounder_latent(conf)
        logits = self._logits_branch(self._hidden(obs, conf_lat))
        value = self._value_branch(self._hidden(obs, conf_lat))

        self._model = tf.keras.Model(inputs=inputs, outputs=[logits, value])
        self.register_variables(self._model.variables)

    def forward(self, input_dict, state, seq_lens):
        model_out, self._value_out = self._model(input_dict["obs_flat"])
        return model_out, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])

    def _confounder_latent(self, conf):
        lat = layers.Dense(32, activation=activations.tanh, kernel_initializer=initializers.RandomNormal(stddev=1.0))(conf)
        lat = layers.Dense(self.num_outputs, kernel_initializer=initializers.RandomNormal(stddev=0.01))(lat)
        return lat

    def _hidden(self, obs, conf):
        cat = layers.Concatenate()([obs, conf])
        hidden = layers.Dense(128, activation=activations.tanh, kernel_initializer=initializers.RandomNormal(stddev=1.0))(cat)
        cat = layers.Concatenate()([hidden, conf])
        hidden = layers.Dense(256, activation=activations.tanh, kernel_initializer=initializers.RandomNormal(stddev=1.0))(cat)
        cat = layers.Concatenate()([hidden, conf])
        hidden = layers.Dense(32, kernel_initializer=initializers.RandomNormal(stddev=1.0))(hidden)
        cat = layers.Concatenate()([hidden, conf])
        return hidden

    def _logits_branch(self, hidden):
        return layers.Dense(self.num_outputs, kernel_initializer=initializers.RandomNormal(stddev=0.01))(hidden)

    def _value_branch(self, hidden):
        return layers.Dense(1, kernel_initializer=initializers.RandomNormal(stddev=0.01))(hidden)
