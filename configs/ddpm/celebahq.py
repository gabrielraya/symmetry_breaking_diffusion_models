# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Config file for reproducing the results of DDPM on celeba32."""

from configs.default_celeba_configs import get_default_configs


def get_config():
  config = get_default_configs()

  # training
  training = config.training
  config.training.batch_size =  12
  training.sde = 'vpsde'
  training.continuous = False
  training.reduce_mean = True
  training.n_iters = 4000001

  # sampling
  sampling = config.sampling
  sampling.sampler = 'ancestral_sampling'

  # data
  data = config.data
  data.dataset = 'CelebAHQ'
  data.centered = True
  data.tfrecords_path = '/home/username/tensorflow_datasets/downloads/manual/celeba-tfr/' #path to tf records
  data.image_size = 256

  # evaluation
  evaluate = config.eval
  evaluate.checkpoint = 60
  evaluate.batch_size = 20 #12 #00 # 500

  # model
  model = config.model
  model.name = 'ddpm'
  model.scale_by_sigma = False
  model.num_scales = 1000
  model.ema_rate = 0.9999
  model.normalization = 'GroupNorm'
  model.nonlinearity = 'swish'
  model.nf = 128
  model.ch_mult = (1, 1, 2, 2, 4, 4)
  model.num_res_blocks = 2
  model.attn_resolutions = (16,)
  model.resamp_with_conv = True
  model.conditional = True

  # optim
  optim = config.optim
  optim.lr = 2e-5

  return config
