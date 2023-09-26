"""
Library with the samplers and functions related to the sampling processes both forward and backwards
"""
import abc
import torch
import numpy as np
from tqdm import tqdm
from models import utils as mutils

_SAMPLERS = {}


def register_sampler(cls=None, *, name=None):
    """A decorator for registering sampler classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _SAMPLERS:
            raise ValueError(f'Already registered model with name: {local_name}')
        _SAMPLERS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def get_sampler(name):
    return _SAMPLERS[name]


def progressive_encoding(x, diffusion, config, n_samples=15):
    """ Runs a forward process iteratively

    We assume t=0 is already a one-step perturbed image

    Args:
        x: batch tensor NxCXHxW
        diffusion: an object from the DiffusionProcess class
        config: the config settings as set in the ml-collections file
        n_samples: a scalar value representing the number of samples equally distributed over the trajectory
    returns:
        n_samples at times T//15
    """

    xs = []

    # Initial sample - sampling from given state
    x = x.to(config.device)

    # equally subsample noisy states
    indx = np.linspace(0, diffusion.T - 1, n_samples, dtype=int)

    with torch.no_grad():
        # time partition [0,T]
        timesteps = torch.arange(0, diffusion.T, device=config.device)

        for t in timesteps:
            t_vec = torch.ones(x.shape[0], dtype=torch.int64, device=t.device) * t
            x, noise = diffusion.forward_step(x, t_vec)
            if t.item() in indx:
                xs.append(x)

        xs = torch.stack(xs, dim=1)

    return xs


class Sampler(abc.ABC):
    """The abstract class for a predictor algorithm."""

    def __init__(self, diffusion, model_fn):
        super().__init__()
        self.diffusion = diffusion
        self.model_fn = model_fn

    @abc.abstractmethod
    def update_fn(self, x, t):
        """One update of the sampler.

        Args:
            x: A PyTorch tensor representing the current state
            t: A PyTorch tensor representing the current time step.

        Returns:
            x: A PyTorch tensor of the next state.
            x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
        """
        pass


@register_sampler(name='ancestral_sampling')
class AncestralSampling(Sampler):
    """The ancestral sampler used in the DDPM paper"""

    def __init__(self, diffusion, model_fn):
        super().__init__(diffusion, model_fn)

    def denoise_update_fn(self, x, t):
        diffusion = self.diffusion

        beta = diffusion.discrete_betas.to(t.device)[t.long()]
        std = diffusion.sqrt_1m_alphas_cumprod.to(t.device)[t.long()]

        predicted_noise = self.model_fn(x, t)  # set the model either for training or evaluation
        score = - predicted_noise / std[:, None, None, None]

        x_mean = (x + beta[:, None, None, None] * score) / torch.sqrt(1. - beta)[:, None, None, None]
        noise = torch.randn_like(x)
        x = x_mean + torch.sqrt(beta)[:, None, None, None] * noise
        return x, x_mean

    def update_fn(self, x, t):
        return self.denoise_update_fn(x, t)


def sampling_fn(config, diffusion, model, shape, inverse_scaler, T=None, denoise=True):
    """If T is given then starts from that diffusion time step"""
    model_fn = mutils.get_model_fn(model, train=False)  # get noise predictor model in evaluation mode
    sampler_method = get_sampler(config.sampling.sampler.lower())
    sampler = sampler_method(diffusion, model_fn)

    if T is None:
        T = diffusion.T

    with torch.no_grad():
        # Initial sample - sampling from tractable prior
        x = diffusion.prior_sampling(shape).to(config.device)
        # reverse time partition [T, 0]
        timesteps = torch.flip(torch.arange(0, T, device=config.device), dims=(0,))

        for timestep in tqdm(timesteps):
            t = torch.ones(shape[0], device=config.device) * timestep
            x, x_mean = sampler.denoise_update_fn(x, t)

        return inverse_scaler(x_mean if denoise else x)


def progressive_generation(config, diffusion, model, shape, inverse_scaler, n_samples=12, denoise=True):
    model_fn = mutils.get_model_fn(model, train=False)  # get noise predictor model in evaluation mode
    sampler_method = get_sampler(config.sampling.sampler.lower())
    sampler = sampler_method(diffusion, model_fn)

    with torch.no_grad():
        # sample from the prior distribution
        x = diffusion.prior_sampling(shape).to(config.device)

        # reverse time partition [T, 0]
        timesteps =torch.flip(torch.arange(0, diffusion.T, device=config.device), dims=(0,))

        # equally subsample noisy states
        indx = np.linspace(0, diffusion.T - 1, n_samples, dtype=int)

        # create array of shape n_samples
        xs = torch.zeros((n_samples, *x.shape))
        i=0

        for timestep in tqdm(timesteps):
            t = torch.ones(shape[0], device=config.device) * timestep
            x, x_mean = sampler.denoise_update_fn(x, t)
            if timestep.item() in indx:
                if timestep.item()==999:
                    xs[i] = x_mean if denoise else x
                else:
                    xs[i] = x
                i+=1
    return inverse_scaler(xs), indx


######################### Fast Samplers ########################################
class FastSampler(abc.ABC):
    """The abstract class for a Fast Sampler algorithm."""

    def __init__(self, diffusion, model_fn):
        super().__init__()
        self.diffusion = diffusion
        self.model_fn = model_fn

    @abc.abstractmethod
    def update_fn(self, x, t, t_next):
        """One update of the sampler.

        Args:
            x: A PyTorch tensor representing the current state
            t: A PyTorch tensor representing the current time step.
            t_next: A PyTorch tensor representing the next time step.

        Returns:
            x: A PyTorch tensor of the next state.
            x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
        """
        pass


@register_sampler(name='ddim')
class DDIM(FastSampler):
    """The DDIM sampler"""

    def __init__(self, diffusion, model_fn):
        super().__init__(diffusion, model_fn)

    def update_fn(self, x, t, t_next, eta=0):
        # NOTE: We are producing each predicted x0, not x_{t-1} at timestep t.

        at = self.diffusion.alphas_cumprod[t.long()].to(x.device)[:, None, None, None]
        at_next = self.diffusion.alphas_cumprod[t_next.long()].to(x.device)[:, None, None, None]

        # noise estimation
        et = self.model_fn(x, t.float())

        # predicts x_0 by direct substitution
        x0_t = (x - et * (1 - at).sqrt()) / at.sqrt()

        # noise controlling the Markovia/Non-Markovian property
        sigma_t = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()

        # update using forward posterior q(x_t-1|x_t, x0_t)
        x = at_next.sqrt() * x0_t + (1 - at_next- sigma_t**2).sqrt() * et + sigma_t * torch.randn_like(x)

        return x, x0_t


################################################ Utils for PSDM sampler #########################################
def transfer(x, t, t_next, et, alphas_cump):
    at = alphas_cump[t.long() + 1].view(-1, 1, 1, 1)
    at_next = alphas_cump[t_next.long() + 1].view(-1, 1, 1, 1)

    x_delta = (at_next - at) * ((1 / (at.sqrt() * (at.sqrt() + at_next.sqrt()))) * x - \
                                1 / (at.sqrt() * (((1 - at_next) * at).sqrt() + ((1 - at) * at_next).sqrt())) * et)

    x_next = x + x_delta
    return x_next


def runge_kutta(x, t_list, model, alphas_cump, ets):
    """

    """
    e_1 = model(x, t_list[0])
    ets.append(e_1)
    x_2 = transfer(x, t_list[0], t_list[1], e_1, alphas_cump)

    e_2 = model(x_2, t_list[1])
    x_3 = transfer(x, t_list[0], t_list[1], e_2, alphas_cump)

    e_3 = model(x_3, t_list[1])
    x_4 = transfer(x, t_list[0], t_list[2], e_3, alphas_cump)

    e_4 = model(x_4, t_list[2])
    et = (1 / 6) * (e_1 + 2 * e_2 + 2 * e_3 + e_4)

    return et


def gen_order_4(img, t, t_next, model, alphas_cump, ets):
    t_list = [t, (t+t_next)/2, t_next]
    if len(ets) > 2:
        noise_ = model(img, t)
        ets.append(noise_)
        noise = (1 / 24) * (55 * ets[-1] - 59 * ets[-2] + 37 * ets[-3] - 9 * ets[-4])
    else:
        noise = runge_kutta(img, t_list, model, alphas_cump, ets)

    img_next = transfer(img, t, t_next, noise, alphas_cump)
    return img_next


@register_sampler(name='psdm')
class PSDM(FastSampler):
    """The PSDM sampler"""

    def __init__(self, diffusion, model_fn):
        super().__init__(diffusion, model_fn)

    def update_fn(self, x, t, t_next, eta=0):
        alphas_cump = self.diffusion.alphas_cumprod.to(x.device)

        # update
        x = gen_order_4(x, t, t_next,  self.model_fn, alphas_cump, ets)

        # update using forward posterior q(x_t-1|x_t, x0_t)
        # x = at_next.sqrt() * x0_t + (1 - at_next- sigma_t**2).sqrt() * et + sigma_t * torch.randn_like(x)
        #TODO TO BE UPDATED
        return 0


def get_fast_sampler(config, diffusion, model, inverse_scaler, sampler_name="ddim"):
    """ Creates a fast sampling function
        Note that DDPM fast sampler runs DDIM with eta=1.0
    """
    model_fn = mutils.get_model_fn(model, train=False)  # get noise predictor model in evaluation mode
    # sampler_method = get_sampler(config.sampling.sampler.lower())
    if sampler_name == "ddpm":
        sampler_method = get_sampler("ddim")
    else:
        sampler_method = get_sampler(sampler_name)
    sampler = sampler_method(diffusion, model_fn)

    def ddim_sampling_fn(x, seq, seq_next, denoise=True):
        with torch.no_grad():
            # reverse time partition [T, 0]
            for i, j in tqdm(zip(reversed(seq), reversed(seq_next)), total=len(seq)):
                t = (torch.ones(x.shape[0]) * i)
                t_next = (torch.ones(x.shape[0]) * j)
                x, x_mean = sampler.update_fn(x, t, t_next)
            return inverse_scaler(x_mean if denoise else x)

    def ddpm_sampling_fn(x, seq, seq_next, denoise=True):
        with torch.no_grad():
            # reverse time partition [T, 0]
            for i, j in tqdm(zip(reversed(seq), reversed(seq_next)), total=len(seq)):
                t = (torch.ones(x.shape[0]) * i)
                t_next = (torch.ones(x.shape[0]) * j)
                x, x_mean = sampler.update_fn(x, t, t_next, eta=1.0)
            return inverse_scaler(x_mean if denoise else x)

    def pndm_sampling_fn(x, seq, seq_next, denoise=True):
        start = True
        alphas_cump = diffusion.alphas_cumprod.to(config.device)
        ets = []

        with torch.no_grad():
            # reverse time partition [T, 0]
            for i, j in tqdm(zip(reversed(seq), reversed(seq_next)), total=len(seq)):
                t = (torch.ones(x.shape[0]) * i)
                t_next = (torch.ones(x.shape[0]) * j)
                x = gen_order_4(x, t, t_next, model_fn, alphas_cump, ets)
            return inverse_scaler(x if denoise else x)

    if sampler_name == "ddim":
        return ddim_sampling_fn
    elif sampler_name == "ddpm":
        return ddpm_sampling_fn
    elif sampler_name == "psdm":
        return pndm_sampling_fn
    else:
        raise ValueError(f"Sampler name {sampler_name} unknown.")


def get_deterministic_trajectories(config, diffusion, model):
    """Run DDIM sampler"""
    model_fn = mutils.get_model_fn(model, train=False)  # get noise predictor model in evaluation mode
    sampler_method = DDIM
    sampler = sampler_method(diffusion, model_fn)

    def ddim_sampling_fn(x, seq, seq_next, denoise=True):
        trajectories=torch.zeros((len(seq)+1,*x.shape))

        # set initial sample
        trajectories[0,:,:,:,:] = x.clone()
        n=1
        with torch.no_grad():
            # reverse time partition [T, 0]
            for i, j in tqdm(zip(reversed(seq), reversed(seq_next)), total=len(seq)):
                t = (torch.ones(x.shape[0]) * i)
                t_next = (torch.ones(x.shape[0]) * j)
                x, x_mean = sampler.update_fn(x, t, t_next)

                if n==len(seq):
                    trajectories[n,:,:,:,:] = x_mean.clone()
                else:
                    trajectories[n,:,:,:,:] = x.clone()

                n+=1

            # return inverse_scaler(x_mean if denoise else x)
            return x_mean, trajectories

    return ddim_sampling_fn
