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

import argparse
import functools
import os
import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent.parent))

import ruamel.yaml as yaml
import tensorflow as tf

from plan2explore import tools
from plan2explore import training
from plan2explore.scripts import configs

os.environ["MUJOCO_GL"] = "osmesa"
os.environ["PYOPENGL_PLATFORM"] = "osmesa"


def process(logdir, rolloutdir, args):
  with args.params.unlocked:
    args.params.logdir = logdir
    args.params.rolloutdir = rolloutdir
  config = configs.make_config(args.params)
  logdir = pathlib.Path(logdir)
  metrics = tools.Metrics(logdir / 'metrics', workers=5)
  training.utility.collect_initial_episodes(metrics, config)
  tf.reset_default_graph()
  dataset = tools.numpy_episodes.numpy_episodes(
      config.train_dir, config.test_dir, config.batch_shape,
      reader=config.data_reader,
      loader=config.data_loader,
      config = config,
      num_chunks=config.num_chunks,
      preprocess_fn=config.preprocess_fn,
      gpu_prefetch=config.gpu_prefetch)
  metrics = tools.InGraphMetrics(metrics)
  build_graph = tools.bind(training.define_model, logdir, metrics)
  for score in training.utility.train(build_graph, dataset, logdir, config):
    yield score


def main(args):
  experiment = training.Experiment(
      args.logdir,
      args.rolloutdir,
      process_fn=functools.partial(process, args=args),
      num_runs=args.num_runs,
      ping_every=args.ping_every,
      resume_runs=args.resume_runs)
  for run in experiment:
    for unused_score in run:
      pass


if __name__ == '__main__':
  boolean = lambda x: bool(['False', 'True'].index(x))
  parser = argparse.ArgumentParser()
  parser.add_argument('--logdir', type=pathlib.Path, default='./logdir/')
  parser.add_argument('--rolloutdir', type=pathlib.Path, default='./rolloutdir/')
  parser.add_argument('--params', default='{}')
  parser.add_argument('--num_runs', type=int, default=1)
  parser.add_argument('--expID', type=str, required=True)
  parser.add_argument('--ping_every', type=int, default=0)
  parser.add_argument('--resume_runs', type=boolean, default=True)
  parser.add_argument('--dmlab_runfiles_path', default=None)
  args_, remaining = parser.parse_known_args()

  args_.params += ' '
  for tmp in remaining:
      args_.params += tmp+' '

  params_ = args_.params.replace('#', ',').replace('\\', '')
  args_.params = tools.AttrDict(yaml.safe_load(params_))
  if args_.dmlab_runfiles_path:
    with args_.params.unlocked:
      args_.params.dmlab_runfiles_path = args_.dmlab_runfiles_path
    assert args_.params.dmlab_runfiles_path  # Mark as accessed.
  args_.logdir = args_.logdir and os.path.expanduser(args_.logdir)
  args_.rolloutdir = args_.rolloutdir and os.path.expanduser(args_.rolloutdir)
  expid = args_.expID.split('_')
  num, comm = int(expid[0]), expid[1:]
  comment = ''
  for com in comm:
      comment += '_'+com

  args_.logdir = os.path.join(args_.logdir, '{:05}_expID'.format(num)+comment)
  args_.rolloutdir = os.path.join(args_.rolloutdir, '{:05}_expID'.format(num)+comment)
  remaining.insert(0, sys.argv[0])
  tf.app.run(lambda _: main(args_), remaining)
