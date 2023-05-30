"""Config file for reproducing the results of DDPM on cifar-10."""

from configs.default_cifar10_configs import get_default_configs


def get_config():
  config = get_default_configs()

  # training
  training = config.training
  training.reduce_mean = True

  # sampling
  sampling = config.sampling
  sampling.sampler = 'ancestral_sampling'

  # data
  data = config.data
  data.centered = True

  # model
  model = config.model
  model.name = 'ddpm'
  model.scale_by_sigma = False
  model.ema_rate = 0.9999
  model.normalization = 'GroupNorm'
  model.nonlinearity = 'swish'
  model.nf = 128
  model.ch_mult = (1, 2, 2, 2)
  model.num_res_blocks = 2
  model.attn_resolutions = (16,)
  model.resamp_with_conv = True
  model.conditional = True

  return config
