import tensorflow as tf
from tensorflow.keras import activations, initializers, regularizers
from tensorflow.keras.layers import Layer


class Condenser(Layer):
    def __init__(self,
                 n_sample_points=15,
                 sampling_bounds=(-1, 100),
                 attention_dim=1,
                 reducer_dim=None,
                 reducer_trainable=False,
                 theta_trainable=True,
                 attention_initializer="glorot_uniform",
                 bias_initializer="zeros",
                 attention_regularizer=None,
                 theta_regularizer=None,
                 bias_regularizer=None,
                 reducer_regularizer=None,
                 attention_activation="leaky_relu",
                 residual_activation=None,
                 reducer_activation=None,
                 use_residual=False,
                 use_reducer=True,
                 scalers_trainable=True,
                 **kwargs):
        super().__init__(**kwargs)

        self.n_sample_points = n_sample_points
        self.sampling_bounds = sampling_bounds
        self.attention_dim = attention_dim
        self.reducer_dim = reducer_dim
        self.reducer_trainable = reducer_trainable
        self.theta_trainable = theta_trainable
        self.scalers_trainable = scalers_trainable

        self.attention_initializer = initializers.get(attention_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.attention_activation = activations.get(attention_activation)
        self.residual_activation = activations.get(residual_activation)
        self.reducer_activation = activations.get(reducer_activation)

        self.attention_regularizer = regularizers.get(attention_regularizer)
        self.theta_regularizer = regularizers.get(theta_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.reducer_regularizer = regularizers.get(reducer_regularizer)

        self.use_residual = use_residual
        self.use_reducer = use_reducer

    def get_config(self):
        config = super().get_config()
        config.update({
            "n_sample_points": self.n_sample_points,
            "sampling_bounds": self.sampling_bounds,
            "attention_dim": self.attention_dim,
            "reducer_dim": self.reducer_dim,
            "reducer_trainable": self.reducer_trainable,
            "theta_trainable": self.theta_trainable,
            "attention_initializer": initializers.serialize(
                self.attention_initializer),
            "bias_initializer": initializers.serialize(self.bias_initializer),
            "attention_activation": activations.serialize(
                self.attention_activation),
            "residual_activation": activations.serialize(
                self.residual_activation),
            "reducer_activation": activations.serialize(
                self.reducer_activation),
            "attention_regularizer": regularizers.serialize(
                self.attention_regularizer),
            "theta_regularizer": regularizers.serialize(
                self.theta_regularizer),
            "bias_regularizer": regularizers.serialize(self.bias_regularizer),
            "reducer_regularizer": regularizers.serialize(
                self.reducer_regularizer),
            "use_residual": self.use_residual,
            "use_reducer": self.use_reducer
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    @staticmethod
    def get_custom_objects():
        return {'Condenser': Condenser}

    def build(self, input_shape):
        in_dim = input_shape[2]
        att_dim = self.attention_dim
        att_latent_dim = int(in_dim * att_dim)

        self.input_dim = in_dim

        # attention weights
        self.att1 = self.add_weight(shape=[in_dim, att_latent_dim],
                                    name="att1",
                                    initializer=self.attention_initializer,
                                    regularizer=self.attention_regularizer)

        self.att2 = self.add_weight(shape=[att_latent_dim, in_dim],
                                    name="att2",
                                    initializer=self.attention_initializer,
                                    regularizer=self.attention_regularizer)

        # characteristic function sampling points
        self.theta = self.add_weight(
            shape=[1, in_dim, self.n_sample_points], name="theta",
            initializer=tf.random_uniform_initializer(*self.sampling_bounds),
            regularizer=self.theta_regularizer,
            trainable=self.theta_trainable)

        # biases
        self.bias_att1 = self.add_weight(shape=[1, att_latent_dim],
                                         name="bias_att1",
                                         initializer=self.bias_initializer,
                                         regularizer=self.bias_regularizer)

        self.bias_att2 = self.add_weight(shape=[1, in_dim],
                                         name="bias_att2",
                                         initializer=self.bias_initializer,
                                         regularizer=self.bias_regularizer)

        # scalers
        self.scale_char = self.add_weight(shape=[1],
                                          name="scale_char",
                                          initializer="ones",
                                          trainable=self.scalers_trainable)

        self.scale_att = self.add_weight(shape=[1],
                                         name="scale_att",
                                         initializer="ones",
                                         trainable=self.scalers_trainable)

        self.scale_input = self.add_weight(shape=[1],
                                           name="scale_input",
                                           initializer="ones",
                                           trainable=self.scalers_trainable)

        if self.use_reducer:
            reducer_in_shape = 2 * in_dim * self.n_sample_points
            reducer_dim = (
                in_dim if self.reducer_dim is None else self.reducer_dim)
            self.reducer = self.add_weight(
                shape=[reducer_in_shape, reducer_dim],
                name="reducer",
                initializer=initializers.Orthogonal(),
                regularizer=self.reducer_regularizer,
                trainable=self.reducer_trainable)
            self.scale_reducer = self.add_weight(
                shape=[1],
                name="scale_reducer",
                initializer="ones",
                trainable=self.scalers_trainable)
            self.bias_reducer = self.add_weight(
                shape=[reducer_dim],
                name="bias_reducer",
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer)

    def compute_mask(self, _, mask=None):
        return None

    def compute_attention_scores(self, input, mask):
        # compute attention weights for all dimensions
        alpha = self.attention_activation(
            tf.matmul(input, self.att1) + self.bias_att1)
        alpha = self.attention_activation(
            tf.matmul(alpha, self.att2) + self.bias_att2)
        # softmax
        alpha = tf.exp(self.scale_att * alpha)
        # ensure attention weights are null on masked values
        if mask is not None:
            mask = tf.expand_dims(tf.cast(mask, tf.float32), -1)
            alpha *= mask
        alpha_sum = tf.reduce_sum(alpha, axis=1, keepdims=True)
        alpha /= tf.clip_by_value(alpha_sum, 1e-9, float("inf"))
        alpha = tf.expand_dims(alpha, axis=-1)
        return alpha

    def compute_ragged_attention_scores(self, input):
        a = tf.ragged.map_flat_values(tf.matmul, input, self.att1)
        alpha = self.attention_activation(a + self.bias_att1)
        b = tf.ragged.map_flat_values(tf.matmul, alpha, self.att2)
        alpha = self.attention_activation(b + self.bias_att2)
        # softmax
        alpha = tf.exp(self.scale_att * alpha)
        alpha_sum = tf.reduce_sum(alpha, axis=1, keepdims=True)
        alpha /= tf.clip_by_value(alpha_sum, 1e-9, float("inf"))
        alpha = tf.expand_dims(alpha, axis=-1)
        return alpha

    def call(self, input, mask=None):
        if isinstance(input, tf.RaggedTensor):
            alpha = self.compute_ragged_attention_scores(input)
        else:
            alpha = self.compute_attention_scores(input, mask)

        # input *= self.scale_input
        # sample characteristic function
        phi = (tf.expand_dims(input, axis=-1)) * (self.theta * self.scale_char)
        # phi *= self.scale_char
        real = tf.reduce_sum(alpha * tf.cos(phi), axis=1)
        imag = tf.reduce_sum(alpha * tf.sin(phi), axis=1)

        # stack real and imaginary parts
        stack = tf.concat([real, imag], axis=-1)
        stack = tf.reshape(
            stack, (-1, 2*self.input_dim*self.n_sample_points))

        # reducer output dim
        if self.use_reducer:
            stack = self.reducer_activation(
                self.scale_reducer * tf.matmul(stack, self.reducer)
                + self.bias_reducer)

        # concatenate characteristic function and input vector
        if self.use_residual:
            res = self.residual_activation(
                tf.reduce_sum(input * alpha[:, :, :, 0], axis=1))
            stack = tf.concat([stack, res], axis=-1)
        return stack


class MultiHeadCondenser(Layer):
    def __init__(self,
                 n_heads=2,
                 hidden_dim=32,
                 hidden_activation="tanh",
                 hidden_regularizer="l2",
                 n_sample_points=15,
                 sampling_bounds=(-1, 100),
                 attention_dim=1,
                 dropout=0,
                 reducer_dim=None,
                 reducer_trainable=False,
                 theta_trainable=True,
                 attention_initializer="glorot_uniform",
                 bias_initializer="zeros",
                 attention_regularizer=None,
                 theta_regularizer=None,
                 bias_regularizer=None,
                 reducer_regularizer=None,
                 attention_activation="leaky_relu",
                 residual_activation=None,
                 reducer_activation=None,
                 use_residual=False,
                 use_reducer=True,
                 scalers_trainable=True,
                 **kwargs):
        super().__init__(**kwargs)
        self.dropout = dropout
        self.denses = [
            tf.keras.layers.Dense(
                hidden_dim,
                activation=hidden_activation,
                kernel_regularizer=hidden_regularizer)
            for _ in range(n_heads)
        ]
        self.condensers = [
            Condenser(
                n_sample_points=n_sample_points,
                sampling_bounds=sampling_bounds,
                attention_dim=attention_dim,
                reducer_dim=reducer_dim,
                reducer_trainable=reducer_trainable,
                theta_trainable=theta_trainable,
                attention_initializer=attention_initializer,
                bias_initializer=bias_initializer,
                attention_regularizer=attention_regularizer,
                theta_regularizer=theta_regularizer,
                bias_regularizer=bias_regularizer,
                reducer_regularizer=reducer_regularizer,
                attention_activation=attention_activation,
                residual_activation=residual_activation,
                reducer_activation=reducer_activation,
                use_residual=use_residual,
                use_reducer=use_reducer,
                scalers_trainable=scalers_trainable)
            for _ in range(n_heads)
        ]

    def call(self, inputs):
        results = []
        for i in range(len(self.condensers)):
            x = self.denses[i](inputs)
            if self.dropout:
                x = tf.keras.layers.Dropout(self.dropout)(x)
            x = self.condensers[i](x)
            results.append(x)
        return tf.concat(results, axis=-1)


class WeightedAttention(Layer):
    def __init__(
            self,
            hidden_dim=32,
            bias_regularizer=None,
            attention_initializer="glorot_normal",
            attention_activation="leaky_relu",
            attention_regularizer=None,
            **kwargs):

        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim

        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.attention_initializer = initializers.get(attention_initializer)
        self.attention_activation = activations.get(attention_activation)
        self.attention_regularizer = regularizers.get(attention_regularizer)

    def build(self, input_shape):
        dim = int(input_shape[-1])
        self.att1 = self.add_weight(
            name="att1",
            shape=(dim, self.hidden_dim),
            initializer=self.attention_initializer,
            regularizer=self.attention_regularizer)

        self.att2 = self.add_weight(
            name="att2",
            shape=(self.hidden_dim, 1),
            initializer=self.attention_initializer,
            regularizer=self.attention_regularizer)

        self.bias = self.add_weight(
            name="bias",
            shape=[2],
            initializer="zeros",
            regularizer=self.bias_regularizer)

        self.scale = self.add_weight(
            name="scale",
            shape=[1],
            initializer="ones")

    def compute_mask(self, _, mask=None):
        return None

    def compute_attention_scores(self, inp, mask):
        # project and evaluate incoming inputs
        alpha = self.attention_activation(
            tf.matmul(inp, self.att1) + self.bias[0])
        alpha = self.attention_activation(
            tf.matmul(alpha, self.att2) + self.bias[1])
        alpha = tf.exp(self.scale * alpha)
        # ensure attention weights are really zero on masked values
        if mask is not None:
            mask = tf.expand_dims(tf.cast(mask, tf.float32), -1)
            alpha *= mask
            alpha_sum = tf.reduce_sum(alpha, axis=1, keepdims=True)
            alpha /= tf.clip_by_value(alpha_sum, 1e-6, float("inf"))
        return alpha

    def compute_ragged_attention_scores(self, inp):
        a = tf.ragged.map_flat_values(tf.matmul, inp, self.att1)
        alpha = self.attention_activation(a + self.bias[0])
        b = tf.ragged.map_flat_values(tf.matmul, alpha, self.att2)
        alpha = self.attention_activation(b + self.bias[1])
        alpha = tf.exp(self.scale * alpha)
        alpha_sum = tf.reduce_sum(alpha, axis=1, keepdims=True)
        alpha /= tf.clip_by_value(alpha_sum, 1e-6, float("inf"))
        return alpha

    def call(self, inp, mask=None):
        if isinstance(inp, tf.RaggedTensor):
            alpha = self.compute_ragged_attention_scores(inp)
        else:
            alpha = self.compute_attention_scores(inp, mask)
        result = tf.math.reduce_sum(inp * alpha, axis=1, keepdims=False)
        return result

    def get_config(self):
        config = super(WeightedAttention, self).get_config()
        config.update({"hidden_dim": self.hidden_dim,
                       "bias_regularizer": self.bias_regularizer,
                       "attention_initializer": self.attention_initializer,
                       "attention_activation": self.attention_activation,
                       "attention_regularizer": self.attention_regularizer})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
