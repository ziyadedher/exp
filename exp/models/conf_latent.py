import numpy as np

from ray.rllib.models.tf.recurrent_tf_modelv2 import RecurrentTFModelV2
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.policy.rnn_sequencing import add_time_dimension
from ray.rllib.utils import try_import_tf
from tensorflow.keras import activations, initializers, layers

tf = try_import_tf()


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


class LearnConf(RecurrentTFModelV2):
    def __init__(self, *args, **kwargs):
        super(LearnConf, self).__init__(*args, **kwargs)
        self._cell_size = 256

        obs_in = layers.Input(shape=(None, np.product(self.obs_space.shape)))
        act_in = layers.Input(shape=(None, np.product(self.action_space.shape) if self.action_space.shape != () else 1))
        rew_in = layers.Input(shape=(None, ), dtype=tf.float32)
        rew_in_t = tf.expand_dims(rew_in, -1)
        total_obs_in = layers.Concatenate()([obs_in, act_in, rew_in_t])

        state_in_h = tf.keras.layers.Input(shape=(self._cell_size, ))
        state_in_c = tf.keras.layers.Input(shape=(self._cell_size, ))
        seq_in = tf.keras.layers.Input(shape=(), dtype=tf.int32)

        conf_lat, state_h, state_c = self._confounder_latent(total_obs_in, seq_in, state_in_h, state_in_c)
        logits = self._logits_branch(self._hidden(obs_in, conf_lat))
        value = self._value_branch(self._hidden(obs_in, conf_lat))

        self._model = tf.keras.Model(
            inputs=[obs_in, act_in, rew_in, seq_in, state_in_h, state_in_c],
            outputs=[logits, value, state_h, state_c])
        self.register_variables(self._model.variables)

    def forward(self, input_dict, state, seq_lens):
        output, new_state = self.forward_rnn(
            add_time_dimension(input_dict["obs_flat"], seq_lens),
            add_time_dimension(tf.expand_dims(tf.cast(input_dict["prev_actions"], dtype=tf.float32), -1), seq_lens),
            add_time_dimension(input_dict["prev_rewards"], seq_lens),
            state, seq_lens)
        return tf.reshape(output, [-1, self.num_outputs]), new_state

    def forward_rnn(self, obs, act, rew, state, seq_lens):
        model_out, self._value_out, h, c = self._model([obs, act, rew, seq_lens] + state)
        return model_out, [h, c]

    def value_function(self):
        return tf.reshape(self._value_out, [-1])

    def get_initial_state(self):
        return [
            np.zeros(self._cell_size, np.float32),
            np.zeros(self._cell_size, np.float32),
        ]

    def _confounder_latent(self, obs, seq, state_h, state_c):
        lstm_out, state_h, state_c = layers.LSTM(self._cell_size, return_sequences=True, return_state=True)(
            inputs=obs,
            mask=tf.sequence_mask(seq),
            initial_state=[state_h, state_c]
        )

        lat = layers.Dense(128, activation=activations.tanh, kernel_initializer=initializers.RandomNormal(stddev=1.0))(lstm_out)
        lat = layers.Dense(32, activation=activations.tanh, kernel_initializer=initializers.RandomNormal(stddev=1.0))(lat)
        lat = layers.Dense(self.num_outputs, kernel_initializer=initializers.RandomNormal(stddev=0.01))(lat)
        return lat, state_h, state_c

    def _hidden(self, obs, conf_lat):
        cat = layers.Concatenate()([obs, conf_lat])
        hidden = layers.Dense(128, activation=activations.tanh, kernel_initializer=initializers.RandomNormal(stddev=1.0))(cat)
        cat = layers.Concatenate()([hidden, conf_lat])
        hidden = layers.Dense(256, activation=activations.tanh, kernel_initializer=initializers.RandomNormal(stddev=1.0))(cat)
        cat = layers.Concatenate()([hidden, conf_lat])
        hidden = layers.Dense(32, kernel_initializer=initializers.RandomNormal(stddev=1.0))(hidden)
        cat = layers.Concatenate()([hidden, conf_lat])
        return hidden

    def _logits_branch(self, hidden):
        return layers.Dense(self.num_outputs, kernel_initializer=initializers.RandomNormal(stddev=0.01))(hidden)

    def _value_branch(self, hidden):
        return layers.Dense(1, kernel_initializer=initializers.RandomNormal(stddev=0.01))(hidden)
