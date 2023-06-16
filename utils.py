import os
import json
import torch
import numpy as np
import logging
import losses
import importlib.util
import models.utils as mutils
from models.ema import ExponentialMovingAverage



def get_time_step(arr, sigma_n):
    """ Estimates the corresponding time value for a given sigma_n based on fixed values of arr
    Args:
        arr: a tensor representing the alphas_cumprod
        sigma_n: a tensor representing the value we want to get the equivalent time step
    Returns:
        returns the index that is closest to the given value representing the time step
    """
    assert 0 <= sigma_n.all() <= 1.0, "sigma_n should be in [0,1]"

    x = sigma_n.cpu().numpy().reshape(-1,1)
    y = arr.repeat((x.shape[0],1)).numpy()

    diff = np.absolute(y-x)
    return torch.tensor(diff.argmin(axis=1))


def get_time_sequence(denoising_steps=10, T=1000, skip_type="uniform", late_t=None):

    if late_t is None:
        # evenly spaced numbers over a half open interval
        if skip_type == "uniform":
            skip = T // denoising_steps
            seq = np.arange(0, T, skip)
        elif skip_type == "quad":
            seq = (np.linspace(0, np.sqrt(T * 0.8), denoising_steps)** 2)
            seq = [int(s) for s in list(seq)]
        else:
            raise NotImplementedError
    else:
        # evenly spaced numbers over a specified closed interval.
        seq = np.linspace(0, late_t, num=denoising_steps, dtype=int)

    seq_next = [-1] + list(seq[:-1])

    return seq, seq_next


def restore_checkpoint(ckpt_dir, state, device):
    if not os.path.exists(ckpt_dir):
        os.makedirs(os.path.dirname(ckpt_dir), exist_ok=True)
        logging.warning(f"No checkpoint found at {ckpt_dir}. Returned the same state as input")
        return state
    else:
        loaded_state = torch.load(ckpt_dir, map_location=device)
        state['optimizer'].load_state_dict(loaded_state['optimizer'])
        state['model'].load_state_dict(loaded_state['model'], strict=False)
        state['ema'].load_state_dict(loaded_state['ema'])
        state['step'] = loaded_state['step']
        return state


def save_checkpoint(ckpt_dir, state):
    saved_state = {
        'optimizer': state['optimizer'].state_dict(),
        'model': state['model'].state_dict(),
        'ema': state['ema'].state_dict(),
        'step': state['step']
    }
    torch.save(saved_state, ckpt_dir)


def load_diffusion_model(config, workdir):
    """ Loads the trained diffusion model
    Args:
        config: Configuration to use.
        workdir: Working directory for checkpoints.
    """

    # Initialize model
    diffusion_model, model_name = mutils.create_model(config)
    optimizer = losses.get_optimizer(config, diffusion_model.parameters())
    ema = ExponentialMovingAverage(diffusion_model.parameters(), decay=config.model.ema_rate)
    state = dict(optimizer=optimizer, model=diffusion_model, ema=ema, step=0)

    # Load model
    checkpoint_dir = os.path.join(workdir, "checkpoints")
    ckpt = config.eval.checkpoint
    logging.info("Evaluation checkpoint: %d" % (ckpt))
    ckpt_filename = os.path.join(checkpoint_dir, "checkpoint_{}.pth".format(ckpt))
    logging.info("Evaluation checkpoint file: " +  ckpt_filename)
    ckpt_path = os.path.join(checkpoint_dir, f'checkpoint_{ckpt}.pth')
    state = restore_checkpoint(ckpt_path, state, device=config.device)
    ema.copy_to(diffusion_model.parameters())
    # model_fn = mutils.get_model_fn(diffusion_model, train=False)
    return diffusion_model, state


# help function to load config file
class Args():
    def __init__(self,dataset, config_dir=None):
        if config_dir is None:
            self.config_dir="../configs/ddpm/{}.py".format(dataset)
        else:
            self.config_dir = config_dir
        self.workdir="./results/ddpm_{}".format(dataset)
        self.mode="train"


def load_config_file(dataset="fashion", config_dir=None):
    FLAGS = Args(dataset, config_dir)
    spec = importlib.util.spec_from_file_location("config_file", FLAGS.config_dir)
    foo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(foo)
    config=foo.get_config()
    return config


# Python program to store list to JSON file
def write_list(results_dir, a_list, file_name="fids"):
    print("Started writing list data into a json file")
    with open(os.path.join(results_dir,'{}.json'.format(file_name)), "w") as fp:
        json.dump(a_list, fp)
        print("Done writing JSON data into .json file")


# Read list to memory
def read_list(results_dir, file_name):
    # for reading also binary mode is important
    with open(os.path.join(results_dir,'{}.json'.format(file_name)), 'r') as fp:
        n_list = json.load(fp)
        return n_list