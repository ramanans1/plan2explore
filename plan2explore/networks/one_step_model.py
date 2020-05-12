from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd

from plan2explore import tools


def one_step_model(state, prev_action, data_shape, model_width_factor, max_objective=False, dist='deterministic'):

    num_layers=2
    activation=tf.nn.relu
    units=data_shape[0]*model_width_factor
    state = tf.stop_gradient(state)
    prev_action = tf.stop_gradient(prev_action)
    inputs = tf.concat([state, prev_action], -1)
    for _ in range(num_layers):
        hidden = tf.layers.dense(inputs, units, activation )
        inputs = tf.concat([hidden, prev_action], -1)

    mean = tf.layers.dense(inputs, int(np.prod(data_shape)), None)
    mean = tf.reshape(mean, tools.shape(state)[:-1] + data_shape)

    if max_objective:
      min_std=1e-2
      init_std=1.0
      std = tf.layers.dense(inputs, int(np.prod(data_shape)), None)
      init_std = np.log(np.exp(init_std) - 1)
      std = tf.nn.softplus(std + init_std) + min_std
      std = tf.reshape(std, tools.shape(state)[:-1] + data_shape)
      dist = tfd.Normal(mean, std)
      dist = tfd.Independent(dist, len(data_shape))
    else:
      dist = tfd.Deterministic(mean)
      dist = tfd.Independent(dist, len(data_shape))

    return dist
