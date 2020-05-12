# Copyright 2019 The Dreamer Authors. Copyright 2020 Plan2Explore Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd

from plan2explore import tools


def encoder(obs, encoder_feature_shape):

  sh = 128 if encoder_feature_shape==512 else 256
  kwargs = dict(strides=2, activation=tf.nn.relu)
  hidden = tf.reshape(obs['image'], [-1] + obs['image'].shape[2:].as_list())
  hidden = tf.layers.conv2d(hidden, 32, 4, **kwargs)
  hidden = tf.layers.conv2d(hidden, 64, 4, **kwargs)
  hidden = tf.layers.conv2d(hidden, 128, 4, **kwargs)
  hidden = tf.layers.conv2d(hidden, sh, 4, **kwargs)
  hidden = tf.layers.flatten(hidden)
  if encoder_feature_shape!=512:
      assert hidden.shape[1:].as_list() == [1024], hidden.shape.as_list()
  else:
      assert hidden.shape[1:].as_list() == [512], hidden.shape.as_list()
  hidden = tf.reshape(hidden, tools.shape(obs['image'])[:2] + [
      np.prod(hidden.shape[1:].as_list())])
  return hidden


def decoder(features, data_shape, std=1.0):
  kwargs = dict(strides=2, activation=tf.nn.relu)
  hidden = tf.layers.dense(features, 1024, None)
  hidden = tf.reshape(hidden, [-1, 1, 1, hidden.shape[-1].value])
  hidden = tf.layers.conv2d_transpose(hidden, 128, 5, **kwargs)
  hidden = tf.layers.conv2d_transpose(hidden, 64, 5, **kwargs)
  hidden = tf.layers.conv2d_transpose(hidden, 32, 6, **kwargs)
  mean = tf.layers.conv2d_transpose(hidden, data_shape[-1], 6, strides=2)
  assert mean.shape[1:].as_list() == data_shape, mean.shape
  mean = tf.reshape(mean, tools.shape(features)[:-1] + data_shape)
  return tfd.Independent(tfd.Normal(mean, std), len(data_shape))
