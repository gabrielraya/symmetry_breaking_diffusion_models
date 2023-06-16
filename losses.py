""" All functions related to loss computation and optimization. """

import torch
import torch.optim as optim
import numpy as np
from models import utils as mutils


def get_optimizer(config, params):
    """Returns a flax optimizer object based on `config`."""
    if config.optim.optimizer == 'Adam':
        optimizer = optim.Adam(params, lr=config.optim.lr, betas=(config.optim.beta1, 0.999), eps=config.optim.eps,
                               weight_decay=config.optim.weight_decay)
    else:
        raise NotImplementedError(
            f'Optimizer {config.optim.optimizer} not supported yet!')
    return optimizer


def optimization_manager(config):
    """Returns an optimize_fn based on `config`."""

    def optimize_fn(optimizer, params, step, lr=config.optim.lr,
                    warmup=config.optim.warmup,
                    grad_clip=config.optim.grad_clip):
        """Optimizes with warmup and gradient clipping (disabled if negative)."""
        if warmup > 0:
            for g in optimizer.param_groups:
                g['lr'] = lr * np.minimum(step / warmup, 1.0)
        if grad_clip >= 0:
            torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
        optimizer.step()

    return optimize_fn


def get_ddpm_loss_fn(diffusion, train=True):
    """ Implements DDPM loss """

    def loss_fn(model, batch):
        model_fn = mutils.get_model_fn(model, train=train) # set model either in train of evaluation mode
        time_steps = torch.randint(0, diffusion.T, (batch.shape[0],), device=batch.device)
        sqrt_alphas_cumprod = diffusion.sqrt_alphas_cumprod.to(batch.device)
        sqrt_1m_alphas_cumprod = diffusion.sqrt_1m_alphas_cumprod.to(batch.device)

        noise = torch.randn_like(batch)
        perturbed_data = sqrt_alphas_cumprod[time_steps, None, None, None] * batch + sqrt_1m_alphas_cumprod[time_steps, None, None, None] * noise


        predicted_noise = model_fn(perturbed_data, time_steps)
        losses = torch.square(predicted_noise - noise)
        loss = torch.mean(losses)
        return loss
    return loss_fn


def get_step_fn(diffusion, train, optimize_fn=None):
    """Create a one-step training/evaluation function.

    Args:
        diffusion: A `diffusion_lib.DiffusionProcess` object that represents the forward Diffusion Process.
        optimize_fn : An optimization function.
    Returns:
        A one-step function for training or evaluation.
    """
    loss_fn = get_ddpm_loss_fn(diffusion, train)

    def step_fn(state, batch):
        """Running one step of training or evaluation.

        This function will undergo `jax.lax.scan` so that multiple steps can be pmapped and jit-compiled together
        for faster execution.

        Args:
            state: A dictionary of training information, containing the score model,
                    optimizer, EMA status, and number of optimization steps.
            batch: A mini-batch of training/evaluation data.

        Returns:
            loss: The average loss value of this state.
        """
        model = state['model']

        if train:
            optimizer = state['optimizer']
            optimizer.zero_grad()
            loss = loss_fn(model, batch)
            loss.backward()
            optimize_fn(optimizer, model.parameters(), step=state['step'])
            state['step'] += 1
            state['ema'].update(model.parameters())
        else:
            with torch.no_grad():
                ema = state['ema']
                ema.store(model.parameters())
                ema.copy_to(model.parameters())
                loss = loss_fn(model, batch)
                ema.restore(model.parameters())
        return loss

    return step_fn