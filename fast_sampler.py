import glob
import gc
import os
import torch
import logging
import numpy as np
import torchvision.utils as vutils
from torchvision.utils import make_grid
from torch.distributions.multivariate_normal import MultivariateNormal

import utils
import evaluation
import sampling
import datasets
import plots as plts

from models import utils as mutils
from sampling import progressive_generation
from diffusion_lib import GaussianDiffusion
from inception import InceptionV3
from utils import load_diffusion_model
from fast_fid import get_batch_stats, calculate_frechet_distance

import io
import tensorflow as tf
import fast_fid as ffid

import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt

# Set environment parameters and visual styles
matplotlib.rcParams['pdf.fonttype'] = 42
os.environ["MKL_NUM_THREADS"] = "1"  # to fix issue with scipy for linalg evaluation for FID when updating packages
sns.set_theme()


def generate_xt_samples(batch, scaler, diffusion, t):
    """Helper function to generate x_t samples."""
    t = t * torch.ones(batch.shape[0], dtype=int, device=batch.device)
    mu, std = diffusion.t_step_transition_prob(scaler(batch), t)
    z = torch.randn_like(batch)
    return mu + std[:, None, None, None]*z


def get_approx_stats(train_ds, diffusion, t, config):
    """Estimates mu and cov for a given noise state x_t at time over the dataset train_ds"""

    t = min(t, 999)
    scaler = datasets.get_data_scaler(config)
    device = config.device

    # make sure we evaluate around 50,000 samples
    num_sampling_rounds = config.eval.num_samples // config.eval.batch_size
    means = np.array([])

    with torch.no_grad():
        for batch_id, batch_dict in enumerate(train_ds):

            # makes sure max num evaluations ~ config.eval.num_samples
            if batch_id >= num_sampling_rounds:
                break;

            batch = torch.from_numpy(batch_dict['image']._numpy()).to(device).float()
            batch = batch.permute(0, 3, 1, 2).to(device)

            # obtain x_t
            x_t_samples = generate_xt_samples(batch, scaler, diffusion, t)
            x_t = x_t_samples.reshape(x_t_samples.shape[0], -1).cpu().numpy()

            means = x_t if means.size == 0 else np.append(means, x_t, axis=0)

            if batch_id % 20 == 0:
                logging.info("Making Gaussian approximation stats -- step: %d" % (batch_id))

        mu = np.mean(means, axis=0)
        cov = np.cov(means, rowvar=False).astype(np.float32)

    return mu, cov


def gauss_approx_stats(config, stats_dir="assets/stats_gauss_approx", t_start=None):
    """ Create Gaussian approximation statistics file containing the mu and Sigma for FID scores..
    Args:
      config: Configuration to use.
      fid_dir: The subfolder for storing fid statistics.
      t_start: Time step for which to compute the Gaussian approximation statistics. If None, runs for all t-values in the list.
    """

    # Create directory to save data stats
    os.makedirs(stats_dir, exist_ok=True)

    # Build data pipeline on evaluation mode
    train_ds, _, _ = datasets.get_dataset(config, uniform_dequantization=False, evaluation=True)
    logging.info("Batch size N -- : %d -- Eval Samples %d -- image size: %d" % (
    config.eval.batch_size, config.eval.num_samples, config.data.image_size))

    # Loads Diffusion Model and Diffusion Process
    diffusion = GaussianDiffusion(config.model.beta_min, config.model.beta_max, T=1000)

    mus = []
    covs = []

    t_values = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000] if t_start is None else [t_start]

    for t in t_values:
        logging.info("Making Gaussian approximation T -- : %d" % (t))
        if config.data.image_size == 256:
            mu, cov = get_approx_stats_HQ(train_ds, diffusion, t, config)
        else:
            mu, cov = get_approx_stats(train_ds, diffusion, t, config)
        mus.append(mu)
        covs.append(cov)

    mus = np.stack(mus)
    covs = np.stack(covs)

    # Save data distribution moments mu and conv for FID scores
    if t_start is None:
        filename = f'{config.data.dataset.lower()}_{config.data.image_size}_stats.npz'
    else:
        filename = f'{config.data.dataset.lower()}_{config.data.image_size}_{t_start}_stats.npz'

    with tf.io.gfile.GFile(os.path.join(stats_dir, filename), "wb") as fout:
        io_buffer = io.BytesIO()
        np.savez_compressed(io_buffer, mu=mus, cov=covs)
        fout.write(io_buffer.getvalue())


def get_approx_stats_HQ(train_ds, diffusion, t, config):
    """
    Approximates the mean and std for Gaussian approximation at a given diffusion time t for HQ datasets
    """

    t = min(t, 999)

    scaler = datasets.get_data_scaler(config)
    device = config.device

    C, H, W = config.data.num_channels, config.data.image_size, config.data.image_size
    centered_sum = torch.zeros((C * H * W, C * H * W), dtype=torch.float32, device=device)
    mean_sum = torch.zeros((C, H, W), dtype=torch.float32)

    # make sure we evaluate around 50,000 samples
    num_sampling_rounds = config.eval.num_samples // config.eval.batch_size

    # tracks total number of samples
    n_samples = 0

    # generate noisy samples
    with torch.no_grad():
        for batch_id, batch_dict in enumerate(train_ds):

            # makes sure max num evaluations ~ config.eval.num_samples
            if batch_id >= num_sampling_rounds:
                break;

            batch = torch.from_numpy(batch_dict['image']._numpy()).to(device).float()
            batch = batch.permute(0, 3, 1, 2)

            # obtain x_t
            x_t = generate_xt_samples(batch, scaler, diffusion, t)

            # compute mean of images
            mean = torch.mean(x_t, dim=0)
            centered = x_t - mean

            # accumulate outer product of centered data
            n = centered.shape[0]
            centered = centered.reshape(n, -1)  # flatten each image
            outer = torch.matmul(centered.T, centered)
            centered_sum += outer

            # accumulate mean
            mean_sum += torch.sum(x_t, dim=0)

            # update sample count
            n_samples += n

            if batch_id % 20 == 0:
                logging.info("Making Gaussian approximation stats -- step: %d" % (batch_id))

    # compute covariance from accumulated centered data
    cov = centered_sum / (n_samples - 1)
    mean = mean_sum / n_samples

    # add regularization to covariance matrix to get positive definite matrix
    lam = 0.1
    reg = lam * torch.eye(cov.shape[0], dtype=cov.dtype, device=cov.device)
    cov += reg

    return mean, cov


def evaluate_fids(config, workdir, eval_folder="eval"):
    """Evaluate trained models starting at different diffusion steps.

    Args:
        config: Configuration to use.
        workdir: Working directory for checkpoints.
        eval_folder: The subfolder for storing evaluation results.
                     Default to "eval".
    """

    # Create directory to eval_folder
    eval_dir = os.path.join(workdir, eval_folder)
    os.makedirs(eval_dir, exist_ok=True)

    # creates results directory
    ckpt = config.eval.checkpoint
    results_dir = os.path.join(eval_dir, "fids_ckpt_{}".format(ckpt))
    os.makedirs(results_dir, exist_ok=True)

    # Create data inverse scaler
    inverse_scaler = datasets.get_data_inverse_scaler(config)

    # loads diffusion model
    diffusion_model, state = load_diffusion_model(config, workdir)

    # define the diffusion model
    diffusion = GaussianDiffusion(config.model.beta_min, config.model.beta_max, T=1000)

    # Load pre-computed dataset statistics.
    data_stats = evaluation.load_dataset_stats(config)
    mu_w, cov_w = data_stats['mu'], data_stats['cov']

    # Load inception network
    incept = InceptionV3().to(config.device)
    incept = incept.eval()

    # run ablation study
    Ts = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

    # sampling
    sampling_shape = (config.eval.batch_size, config.data.num_channels,
                      config.data.image_size, config.data.image_size)

    # creates a new directory per T for each FID evaluation
    fids = []
    for T in Ts:
        logging.info("sampling starting at T={}".format(T))
        # save Inception network activations
        activations = np.array([])
        sampling_plots = True
        num_sampling_rounds = config.eval.num_samples // config.eval.batch_size + 1
        for r in range(num_sampling_rounds):
            logging.info("sampling -- ckpt: %d, round: %d" % (ckpt, r))
            # Directory to save samples. Different for each host to avoid writing conflicts
            this_sample_dir = os.path.join(results_dir, f"time_t_{T}")
            tf.io.gfile.makedirs(this_sample_dir)
            samples = sampling.sampling_fn(config, diffusion, diffusion_model, sampling_shape, inverse_scaler, T=T,
                                           denoise=True)
            samples = np.clip(samples.cpu().numpy() * 1., 0, 1).astype(np.float32)
            if sampling_plots:
                logging.info("Saving generated samples at {}".format(this_sample_dir))
                plts.save_image(torch.tensor(samples), this_sample_dir, n=64, pos="vertical", padding=0,
                                name="{}_{}_data_samples_T_{}".format(config.model.name, config.data.dataset.lower(),
                                                                      T))
                sampling_plots = False
            # samples = samples.reshape((-1, config.data.image_size, config.data.image_size, config.data.num_channels))
            # Write samples to disk
            # with tf.io.gfile.GFile(
            #         os.path.join(this_sample_dir, f"samples_{r}.npz"), "wb") as fout:
            #     io_buffer = io.BytesIO()
            #     np.savez_compressed(io_buffer, samples=samples)
            #     fout.write(io_buffer.getvalue())

            # Force garbage collection before calling TensorFlow code for Inception network
            gc.collect()

            # get inception logits
            if samples.shape[1] == 1:
                # HACK: Inception expects three channels so we tile
                batch = torch.tensor(samples).repeat((1, 3, 1, 1)).numpy()
            else:
                batch = samples
            # get inception logits
            batch_activations = ffid.get_batch_activations(torch.tensor(batch).to(config.device), model=incept)
            activations = batch_activations if activations.size == 0 else np.append(activations, batch_activations,
                                                                                    axis=0)
            # Force garbage collection again before returning logits
            gc.collect()

        # generative approximation distribution moments
        mu, cov = ffid.get_statistics_numpy(activations)

        # Save data distribution moments mu and conv for FID scores
        filename = f'{config.data.dataset.lower()}_{config.data.image_size}_stats.npz'
        with tf.io.gfile.GFile(os.path.join(this_sample_dir, filename), "wb") as fout:
            io_buffer = io.BytesIO()
            np.savez_compressed(io_buffer, mu=mu, cov=cov)
            fout.write(io_buffer.getvalue())

        # Compute FID on all samples together.
        fid = ffid.calculate_frechet_distance(mu, cov, mu_w, cov_w)
        fids.append(fid)

        del mu, cov
        logging.info("T-%d --- FID: %.6e" % (T, fid))

    fig = plt.figure(figsize=(7, 4))
    plt.plot(Ts, fids)
    plt.scatter(Ts, fids)
    plt.xlabel("Diffusion time steps $t$")
    plt.ylabel("FID")
    fig.savefig(os.path.join(results_dir, "late_start_fids.pdf"), dpi=fig.dpi, bbox_inches='tight')
    fig.savefig(os.path.join(results_dir, "late_start_fids.png"), dpi=fig.dpi, bbox_inches='tight')
    plt.close(fig)

    # save FIDs
    utils.write_list(results_dir, fids)


def evaluate_fids_fast_samplers(config, workdir, eval_folder="eval",
                                n_steps=10, sampler_name="ddim", skip_type="uniform", gaussian_approximation=False):
    """Evaluate trained models using a fast sampler starting at different diffusion steps.

    Args:
        config: Configuration to use.
        workdir: Working directory for checkpoints.
        eval_folder: The subfolder for storing evaluation results.
                     Default to "eval".
        sampler_name: fast sampler name to use ["ddim", "distillation", "psdm"]
        gaussian_approximation: if `True` approximates distribution for x_t
    """

    # Create directory to eval_folder
    eval_dir = os.path.join(workdir, eval_folder)
    os.makedirs(eval_dir, exist_ok=True)

    # creates results directory
    ckpt = config.eval.checkpoint

    if gaussian_approximation:
        project_name = "Fast_samplers_ls_gauss_approx"
        # train_ds, eval_ds, _ = datasets.get_dataset(config, uniform_dequantization=False, evaluation=True)
        results_dir = os.path.join(eval_dir,
                                   "{}_fids_ckpt_{}_n_steps_{}_gauss_approx".format(sampler_name, ckpt, n_steps))
        stats_dir = "./assets/stats_gauss_approx"
        filename = f'{config.data.dataset.lower()}_{config.data.image_size}_stats.npz'
    else:
        project_name = "Fast_samplers_late_start_fids"
        results_dir = os.path.join(eval_dir, "{}_fids_ckpt_{}_n_steps_{}".format(sampler_name, ckpt, n_steps))

    os.makedirs(results_dir, exist_ok=True)

    # Create data inverse scaler
    inverse_scaler = datasets.get_data_inverse_scaler(config)

    # loads diffusion model
    diffusion_model, state = load_diffusion_model(config, workdir)

    # define the diffusion model
    diffusion = GaussianDiffusion(config.model.beta_min, config.model.beta_max, T=1000)

    # Load pre-computed dataset statistics.
    data_stats = evaluation.load_dataset_stats(config)
    mu_w, cov_w = data_stats['mu'], data_stats['cov']

    # Load inception network
    incept = InceptionV3().to(config.device)
    incept = incept.eval()

    # 🐝 1️⃣ Start a new run to track this script
    wandb.init(
        # Set the project where this run will be logged
        project=project_name,
        # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
        name="{}_{}_n_steps_{}_{}_data_samples".format(sampler_name, n_steps, config.model.name,
                                                       config.data.dataset.lower()),
        # Track hyperparameters and run metadata
        config=config
    )

    # 🐝 Create a wandb Table to log images, labels and predictions to
    table = wandb.Table(columns=["image", "T", "fid_score", "n_steps", "batch_size", "Image Size", "seq", "seq_next"])

    # run ablation study
    Ts = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 999]

    if sampler_name == "psdm":
        Ts = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 998]
    # Ts=500
    # sampling
    sampling_shape = (config.eval.batch_size, config.data.num_channels,
                      config.data.image_size, config.data.image_size)

    # get sampler function
    sampling_fn = sampling.get_fast_sampler(config, diffusion, diffusion_model, inverse_scaler, sampler_name)

    # creates a new directory per T for each FID evaluation
    fids = []
    for i, T in enumerate(Ts):
        logging.info("sampling starting at T={}".format(T))
        # Skipping strategy
        seq, seq_next = utils.get_time_sequence(n_steps, diffusion.T, skip_type, late_t=T)
        # save Inception network activations
        activations = np.array([])
        sampling_plots = True
        num_sampling_rounds = config.eval.num_samples // config.eval.batch_size + 1

        # fits Gaussian approximation if True
        if gaussian_approximation:
            stats = np.load(os.path.join(stats_dir, filename))
            mu=stats["mu"][i]
            cov=stats["cov"][i]
            # mu = stats["mu"][5]
            # cov = stats["cov"][5]
            dist = MultivariateNormal(torch.tensor(mu), covariance_matrix=torch.tensor(cov))

        # evaluate ~50,000 generated samples
        for r in range(num_sampling_rounds):
            logging.info("sampling -- ckpt: %d, round: %d" % (ckpt, r))
            # Directory to save samples. Different for each host to avoid writing conflicts
            this_sample_dir = os.path.join(results_dir, f"time_t_{T}")
            tf.io.gfile.makedirs(this_sample_dir)

            # use gaussian estimate if True
            if gaussian_approximation:
                x = dist.sample((sampling_shape[0],))
                x = x.reshape((*sampling_shape,)).to(config.device)
            else:
                x = diffusion.prior_sampling(sampling_shape).to(config.device)

            samples = sampling_fn(x, seq, seq_next)
            samples = np.clip(samples.cpu().numpy() * 1., 0, 1).astype(np.float32)
            if sampling_plots:
                logging.info("Saving generated samples at {}".format(this_sample_dir))

                if config.data.image_size > 64:
                    plts.save_image(torch.tensor(samples), this_sample_dir, n=36, pos="square", padding=0,
                                    name="{}_{}_{}_data_samples_T_{}_{}_steps".format(config.model.name, sampler_name,
                                                                                      config.data.dataset.lower(), T,
                                                                                      n_steps))
                else:
                    plts.save_image(torch.tensor(samples), this_sample_dir, n=64, pos="square", padding=0,
                                    name="{}_{}_{}_data_samples_T_{}_{}_steps".format(config.model.name, sampler_name,
                                                                                      config.data.dataset.lower(), T,
                                                                                      n_steps))
                sampling_plots = False
            # samples = samples.reshape((-1, config.data.image_size, config.data.image_size, config.data.num_channels))
            # Write samples to disk # commented for now
            with tf.io.gfile.GFile(
                    os.path.join(this_sample_dir, f"samples_{r}.npz"), "wb") as fout:
                io_buffer = io.BytesIO()
                np.savez_compressed(io_buffer, samples=samples)
                fout.write(io_buffer.getvalue())

            # Force garbage collection before calling TensorFlow code for Inception network
            gc.collect()

            # get inception logits
            if samples.shape[1] == 1:
                # HACK: Inception expects three channels so we tile
                batch = torch.tensor(samples).repeat((1, 3, 1, 1)).numpy()
            else:
                batch = samples
            # get inception logits
            batch_activations = ffid.get_batch_activations(torch.tensor(batch).to(config.device), model=incept)
            activations = batch_activations if activations.size == 0 else np.append(activations, batch_activations,
                                                                                    axis=0)

            # Force garbage collection again before returning logits
            gc.collect()

            # break; # uncomment later #TODO

        # generative approximation distribution moments
        mu, cov = ffid.get_statistics_numpy(activations)
        # Save data distribution moments mu and conv for FID scores
        filename = f'{config.data.dataset.lower()}_{config.data.image_size}_stats.npz'
        with tf.io.gfile.GFile(os.path.join(this_sample_dir, filename), "wb") as fout:
            io_buffer = io.BytesIO()
            np.savez_compressed(io_buffer, mu=mu, cov=cov)
            fout.write(io_buffer.getvalue())
        # Compute FID on all samples together.
        fid = ffid.calculate_frechet_distance(mu, cov, mu_w, cov_w)
        fids.append(fid)
        del mu, cov

        logging.info("T-%d --- FID: %.6e" % (T, fid))

        # 🐝 2️⃣ Log metrics from your script to W&B

        if config.data.image_size > 64:
            sample_grid = make_grid(torch.tensor(samples)[:36], nrow=int(np.sqrt(64)), padding=0)
        else:
            sample_grid = make_grid(torch.tensor(samples)[:64], nrow=int(np.sqrt(64)), padding=0)
        table.add_data(wandb.Image(sample_grid.permute(1, 2, 0).to("cpu").numpy()), T, fid, n_steps,
                       samples.shape[0], str(samples.shape[1:]), str(seq), str(seq_next))

    # close table
    wandb.log({"results_fids_for_different_times_{}".format(sampler_name): table}, commit=False)

    # create custom plot for FID scores
    data = [[x, y] for (x, y) in zip(Ts, fids)]
    table = wandb.Table(data=data, columns=["Diffusion time steps t", "FID"])
    wandb.log({"my_custom_plot_id": wandb.plot.line(table, "Diffusion time steps t", "FID",
                                                    title="Diffusion steps vs FIDs")})

    # 🐝 Close your wandb run
    wandb.finish()

    fig = plt.figure(figsize=(7, 4))
    plt.plot(Ts, fids)
    plt.scatter(Ts, fids)
    plt.xlabel("Diffusion time steps $t$")
    plt.ylabel("FID")
    fig.savefig(os.path.join(results_dir,
                             "{}_{}_late_start_fids_{}_steps.pdf".format(sampler_name, config.data.dataset.lower(),
                                                                         n_steps)), dpi=fig.dpi, bbox_inches='tight')
    fig.savefig(os.path.join(results_dir,
                             "{}_{}_late_start_fids_{}_steps.png".format(sampler_name, config.data.dataset.lower(),
                                                                         n_steps)), dpi=fig.dpi, bbox_inches='tight')
    plt.close(fig)

    # save FIDs
    utils.write_list(results_dir, fids)


def visualize(config, x, Ts, sampling_fn, n_steps, diffusion_T, skip_type,
              results_dir, sampler_name, sizes, samples_ground_truth):
    """Helper function to visualize results for a given sampler and a fixed x~N(0,I)"""

    # creates array to save samples over different late start times
    samples_array = torch.zeros_like(torch.empty(len(Ts) + 1, *x.shape))
    info = []
    # Run generation for different late starting points
    for i, T in enumerate(Ts):
        logging.info("sampling starting at T={}".format(T))
        seq, seq_next = utils.get_time_sequence(n_steps, diffusion_T, skip_type, late_t=T)
        samples = sampling_fn(x, seq, seq_next)
        samples_array[i] = samples

        # save info
        thisdict = dict(time=T, seq=seq.tolist(), seq_next=list(map(int, seq_next)))
        info.append(thisdict)

        # plot grids for each T
        for s in sizes:
            plts.save_image(torch.clip(samples * 255, 0, 255).int(), results_dir, n=s, pos="vertical", padding=0,
                            name="{}_{}_{}_data_samples_T_{}_{}_steps_{}_grid".format(config.model.name, sampler_name,
                                                                                      config.data.dataset.lower(), T,
                                                                                      n_steps, s))
        # plot horizontal for visualization
        for j in [5, 8]:
            plts.save_image(torch.clip(samples * 255, 0, 255).int(), results_dir, n=j, pos="horizontal", padding=0,
                            name="{}_{}_{}_data_samples_{}_images_line_{}_T_{}_steps".format(config.model.name,
                                                                                             sampler_name,
                                                                                             config.data.dataset.lower(),
                                                                                             j, T, n_steps))

    # add ground truth at the end
    samples_array[i + 1] = samples_ground_truth

    # plot progressive visualization
    if x.shape[0] < 16:
        for i in range(x.shape[0]):
            plts.save_image(torch.clip(samples_array[:, i] * 255, 0, 255).int(), results_dir, n=samples_array.shape[0],
                            pos="horizontal", padding=0,
                            name="{}_{}_{}_progressive_T_{}_{}_steps_{}".format(config.model.name, sampler_name,
                                                                                config.data.dataset.lower(), T, n_steps,
                                                                                i))
    else:
        for i in range(16):
            plts.save_image(torch.clip(samples_array[:, i] * 255, 0, 255).int(), results_dir, n=samples_array.shape[0],
                            pos="horizontal", padding=0,
                            name="{}_{}_{}_progressive_T_{}_{}_steps_{}".format(config.model.name, sampler_name,
                                                                                config.data.dataset.lower(), T, n_steps,
                                                                                i))
    # save info
    utils.write_list(results_dir, info, "info")


def visualize_analysis(config, workdir, eval_folder="eval", skip_type="uniform", N=64):
    """ This function provides a visual analysis of the unstable fixed points by running
    1. Evaluate degradation performance for different late starts of DDIM over start 1000, 900, 800, 700, ...
    2. DDIM over 1000 steps -> "ground truth"
    3. Evaluate DDIM on 5, 10 steps late starts
    4. Evaluate PSNM on 5, 10 steps for different late starts
    Visualize behavior of late start in fast samplers for different starting points.

    Args:
        config: Configuration to use.
        workdir: Working directory for checkpoints.
        eval_folder: The subfolder for storing evaluation results.
                     Default to "eval".
        Ts: a list with the late start times to evalute
        sampler_name: fast sampler name to use ["ddim", "distillation", "psdm"]
    """
    # Create directory to eval_folder
    eval_dir = os.path.join(workdir, eval_folder)
    os.makedirs(eval_dir, exist_ok=True)

    # creates results directory
    ckpt = config.eval.checkpoint
    results_dir = os.path.join(eval_dir, "visuals")
    os.makedirs(results_dir, exist_ok=True)

    # Create data inverse scaler
    inverse_scaler = datasets.get_data_inverse_scaler(config)

    # loads diffusion model
    diffusion_model, state = load_diffusion_model(config, workdir)

    # define the diffusion model
    diffusion = GaussianDiffusion(config.model.beta_min, config.model.beta_max, T=1000)

    # sampling
    sampling_shape = (N, config.data.num_channels, config.data.image_size, config.data.image_size)
    sampling_fn = sampling.get_fast_sampler(config, diffusion, diffusion_model, inverse_scaler, "ddim")

    # sample fixed prior
    x = diffusion.prior_sampling(sampling_shape).to(config.device)

    # save images for reproducibility
    # with tf.io.gfile.GFile(
    #         os.path.join(results_dir, f"prior_samples.npz"), "wb") as fout:
    #     io_buffer = io.BytesIO()
    #     np.savez_compressed(io_buffer, samples=x.cpu())
    #     fout.write(io_buffer.getvalue())

    # evaluate for classical deterministic DDPM sampler (DDIM without skipping) for different late start points
    if config.data.image_size >= 256:
        Ts = [400, 500, 600, 700, 800, 900, 999]
    else:
        Ts = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 999]

    for T in Ts:
        seq, seq_next = utils.get_time_sequence(T, diffusion.T, skip_type="uniform", late_t=T)
        samples_ground_truth = sampling_fn(x, seq, seq_next)
        # save plots for different sizes
        plts.save_image(torch.clip(samples_ground_truth * 255, 0, 255).int(), results_dir, n=8, pos="v", nrow=8,
                        padding=0,
                        name="ls_{}_{}_{}_data_samples_T_{}_{}_hor".format(config.model.name, "ddim",
                                                                           config.data.dataset.lower(), T, 8),
                        show=False)

        plts.save_image(torch.clip(samples_ground_truth * 255, 0, 255).int(), results_dir, n=10, pos="v", nrow=10,
                        padding=0,
                        name="ls_{}_{}_{}_data_samples_T_{}_{}_hor".format(config.model.name, "ddim",
                                                                           config.data.dataset.lower(), T, 10),
                        show=False)

        if config.data.image_size <= 32:
            plts.save_image(torch.clip(samples_ground_truth * 255, 0, 255).int(), results_dir, n=12, pos="v", nrow=12,
                            padding=0,
                            name="ls_{}_{}_{}_data_samples_T_{}_{}_hor".format(config.model.name, "ddim",
                                                                               config.data.dataset.lower(), T, 12),
                            show=False)


#         with tf.io.gfile.GFile(
#                 os.path.join(results_dir, f"samples_T_{T}.npz"), "wb") as fout:
#             io_buffer = io.BytesIO()
#             np.savez_compressed(io_buffer, samples=x.cpu())
#             fout.write(io_buffer.getvalue())

#     # Evaluation using Fast Samplers
#     # generates ground truth samples using 1000 denoising steps with DDIM sampler (ODE)
#     # seq, seq_next = utils.get_time_sequence(1000, diffusion.T, skip_type="uniform", late_t=None)
#     # samples_ground_truth = sampling_fn(x, seq, seq_next)

#     # plots over different grids [1^2, 2^2, ..., N]
#     s = lambda x: x**2
#     sizes = s(np.arange(1, int(np.sqrt(N))+1))   # sizes = [4, 9, 16, 25, 36, 64]
#     for s in sizes:
#         plts.save_image(torch.clip(samples_ground_truth * 255, 0, 255).int(), results_dir, n=s, pos="vertical", padding=0,
#                         name="{}_{}_{}_samples_ground_truth_{}_images".format(config.model.name, "ddim", config.data.dataset.lower(), s))

#     # plot horizontal for visualization
#     for i in [5, 8]:
#         plts.save_image(torch.clip(samples_ground_truth * 255, 0, 255).int(), results_dir, n=i, pos="horizontal", padding=0,
#                         name="{}_{}_{}_samples_ground_truth_{}_images_line".format(config.model.name, "ddim", config.data.dataset.lower(), i))

#     # Run evaluation over fast samplers
#     for n_steps in [5, 10]:
#         for sampler_name in ["ddim", "psdm"]:

#             # creates results directory
#             exp_dir = os.path.join(results_dir, "{}_visuals_ckpt_{}_n_steps_{}".format(sampler_name, ckpt, n_steps))
#             os.makedirs(exp_dir, exist_ok=True)

#             # get sampling function according to sampler name
#             sampling_fn = sampling.get_fast_sampler(config, diffusion, diffusion_model, inverse_scaler, sampler_name)

#             # TODO: check why psdm starts at different time
#             # To speed up computation for HQ images
#             if sampler_name == "psdm":
#                 if config.data.image_size >= 256:
#                     Ts = [400, 500, 600, 700, 800, 900, 998]
#                 else:
#                     Ts = [50, 100, 200,300, 400, 500, 600, 700, 800, 900, 998]
#             else:
#                 if config.data.image_size >= 256:
#                     Ts = [400, 500, 600, 700, 800, 900, 999]
#                 else:
#                     Ts = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 999]

#             # run visualizer
#             visualize(config, x, Ts, sampling_fn, n_steps, diffusion.T, skip_type,
#                       exp_dir, sampler_name, sizes, samples_ground_truth)


def evaluate_potential(x, x1, x2, t, diffusion_model, diffusion, alpha, delta_alpha, config):
    # normalized vector
    diff = x2 - x1
    norm = torch.norm(diff)
    v = (diff / norm).to(x.device)

    with torch.no_grad():
        model_fn = mutils.get_model_fn(diffusion_model, train=False)
        t = torch.ones(x.shape[0], device=config.device) * t
        beta_t = diffusion.discrete_betas.to(t.device)[t.long()]
        std = diffusion.sqrt_1m_alphas_cumprod.to(t.device)[t.long()]

        predicted_noise = diffusion_model(x, t)
        score = - predicted_noise / std[:, None, None, None]
        grad_u = -((x + beta_t[:, None, None, None] * score) / torch.sqrt(1. - beta_t)[:, None, None, None] - x)
        # v = 1/(2*torch.sqrt(alpha))[:, None, None, None] *x1  - 1/(2*torch.sqrt(1-alpha))[:, None, None, None] *x2

        v = -torch.sin(alpha)[:, None, None, None] * x1 + torch.cos(alpha)[:, None, None, None] * x2
        v = v.to(config.device)

        potential = np.zeros(len(grad_u))
        cum = 0

        for i in range(len(grad_u)):
            cum += torch.dot(grad_u[i].view(-1), v[i].view(-1) * delta_alpha.item())
            potential[i] = cum

    return potential


def visualize_potentials(config, workdir, eval_folder="eval", generate=False, grid_size=200):
    """ This function provides a visual analysis of the unstable fixed points by running
    1. Generate samples over the DDPM 1000 at different subtimes

    2. DDIM over 1000 steps -> "ground truth"
    3. Evaluate DDIM on 5, 10 steps late starts
    4. Evaluate PSNM on 5, 10 steps for different late starts
    Visualize behavior of late start in fast samplers for different starting points.

    Args:
        config: Configuration to use.
        workdir: Working directory for checkpoints.
        eval_folder: The subfolder for storing evaluation results.
                     Default to "eval".
        Ts: a list with the late start times to evalute
        sampler_name: fast sampler name to use ["ddim", "distillation", "psdm"]
    """
    # Create directory to eval_folder
    eval_dir = os.path.join(workdir, eval_folder)
    os.makedirs(eval_dir, exist_ok=True)

    # creates results directory
    ckpt = config.eval.checkpoint
    results_dir = os.path.join(eval_dir, "potentials")
    os.makedirs(results_dir, exist_ok=True)

    # Create data inverse scaler
    inverse_scaler = datasets.get_data_inverse_scaler(config)

    # loads diffusion model
    diffusion_model, state = load_diffusion_model(config, workdir)

    # define the diffusion model
    diffusion = GaussianDiffusion(config.model.beta_min, config.model.beta_max, T=1000)

    # sampling
    sampling_shape = (config.eval.batch_size, config.data.num_channels, config.data.image_size, config.data.image_size)

    if generate:
        # generates 10 samples at times 1, 100, 200, ,..., 1000
        samples, ts = progressive_generation(config, diffusion, diffusion_model, sampling_shape, inverse_scaler,
                                             n_samples=11)

        # save images for reproducibility
        with tf.io.gfile.GFile(
                os.path.join(results_dir, f"samples.npz"), "wb") as fout:
            io_buffer = io.BytesIO()
            np.savez_compressed(io_buffer, samples=samples.cpu())
            fout.write(io_buffer.getvalue())

    else:
        # load generated samples
        files = glob.glob(os.path.join(results_dir, 'samples_*.npz'))

        # # Load the npz file
        data = np.load(files[0])
        samples = torch.tensor(data["samples"])
        ts = np.linspace(0, diffusion.T - 1, 11, dtype=int)
    # save samples
    plts.save_image(torch.clip(samples[-1] * 255, 0, 255).int(), results_dir, n=64, pos="square", padding=0,
                    name="{}_samples".format(config.data.dataset.lower()), show=False)

    # Plot the potential
    alpha = torch.linspace(-0.2 * torch.pi, 0.5 * torch.pi + 0.2 * torch.pi, 20)

    logging.info("Generating interpolations ")
    # Sanity check of the interpolation over ts
    for i in range(0, len(samples), 1):
        # interpolate over x1 and x2 over ts diffusion times
        x1 = samples[i][6]
        x2 = samples[i][7]
        t = 999 - ts[i]
        x = torch.cos(alpha)[:, None, None, None] * x1 + torch.sin(alpha)[:, None, None, None] * x2
        plts.save_image(torch.clip(x * 255, 0, 255).int(), results_dir, n=len(alpha), w=5, scale=1, padding=0, nrow=20,
                        name="{}_interpolations_{}_{}".format(config.data.dataset.lower(), i, t), pos="vertical",
                        show=False)

    # Compute the potential over interpolated trajectories for different diffusion times

    # Define parameters
    alpha = torch.linspace(-0.2 * torch.pi, 0.5 * torch.pi + 0.2 * torch.pi, grid_size)
    delta_alpha = alpha[1] - alpha[0]

    # Create subplots
    fig, axs = plt.subplots(1, 3, figsize=(15, 4), sharey=False)

    logging.info("Evaluating the potential over alphas")
    # Iterate over desired time steps
    for j, i in enumerate([0, 5, 10]):
        x1 = samples[i][6]
        x2 = samples[i][7]
        t = 999 - ts[i]

        # Calculate x based on alpha
        x = torch.cos(alpha)[:, None, None, None] * x1 + torch.sin(alpha)[:, None, None, None] * x2
        x = x.to(config.device)

        # Calculate potential
        potential = evaluate_potential(x, x1, x2, t, diffusion_model, diffusion, alpha, delta_alpha, config)

        # Plot potential
        ax = axs[j]
        ax.plot(alpha, potential)
        ax.axvline(x=0, ls='--', label=r"$y_1$", color="k", linewidth=1)
        ax.axvline(x=1.5, ls='--', label=r"$y_2$", color="r", linewidth=1)

        # add text centered
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2

        if t == 999:
            ax.text(center_x, center_y, r"$T- t$={}".format(t + 1), fontsize=12, ha='center', va='center')
        else:
            ax.text(center_x, center_y, r"$T- t$={}".format(t), fontsize=12, ha='center', va='center')

        ax.legend()

        ax.set_xlabel(r'$\alpha$')

    # Adjust the spacing between subplots and save the figure
    plt.subplots_adjust(wspace=0.2)
    fig.savefig(os.path.join(results_dir, "potentials_{}.png".format(config.data.dataset.lower())), dpi=200,
                format='png', bbox_inches='tight')

    # Show images
    for img_name, img in zip(
            ["{}_im1".format(config.data.dataset.lower()), "{}_im2".format(config.data.dataset.lower())], [x1, x2]):
        plts.show_image(torch.clip(img * 255, 0, 255).int(), results_dir, name=img_name, w=4, show=False)

    logging.info("Plotting the progressive generation reduced")

    C, H, W = config.data.num_channels, config.data.image_size, config.data.image_size
    # remove the first n_remove=4 samples corresponding to times 1000, 900, 800, 700,
    n_remove = 5
    n_to_show = 2
    n_samples = samples.shape[0] - n_remove

    # reshape the tensor to [12*9, 3, 64, 64]
    i = 6
    tensor = samples[::2, i:i + n_to_show].transpose(0, 1)
    tensor = tensor.reshape(-1, C, H, W)

    # create a grid of images
    grid = vutils.make_grid(tensor, nrow=n_samples, padding=0)
    grid = torch.clip(grid * 255, 0, 255).int()

    # Calculate the desired size for each image
    image_size = H  # Fixed size of each image
    scale_factor = 4  # Adjust this value to increase the image size

    # Calculate the figure size based on the desired image size and scale factor
    fig_width = image_size * n_samples * scale_factor / 100  # Adjust this value as needed
    fig_height = image_size * n_samples * scale_factor / 50  # Adjust this value as needed

    # plot the grid with increased image size
    fig = plt.figure(figsize=(fig_width, fig_height))
    plt.imshow(grid.permute(1, 2, 0).numpy())

    plt.axis('off')
    name = "{}_progressive_generation".format(config.data.dataset.lower())
    fig.savefig(os.path.join(results_dir, "{}.{}".format(name, "png")), bbox_inches='tight', format="png", pad_inches=0)


def visualize_potentials_avg(config, workdir, eval_folder="eval", generate=False, grid_size=200):
    """ This function provides a visual analysis of the unstable fixed points by running
    1. Generate samples over the DDPM 1000 at different subtimes

    2. DDIM over 1000 steps -> "ground truth"
    3. Evaluate DDIM on 5, 10 steps late starts
    4. Evaluate PSNM on 5, 10 steps for different late starts
    Visualize behavior of late start in fast samplers for different starting points.

    Args:
        config: Configuration to use.
        workdir: Working directory for checkpoints.
        eval_folder: The subfolder for storing evaluation results.
                     Default to "eval".
        Ts: a list with the late start times to evalute
        sampler_name: fast sampler name to use ["ddim", "distillation", "psdm"]
    """
    # Create directory to eval_folder
    eval_dir = os.path.join(workdir, eval_folder)
    os.makedirs(eval_dir, exist_ok=True)

    # creates results directory
    ckpt = config.eval.checkpoint
    results_dir = os.path.join(eval_dir, "potentials_avg")
    os.makedirs(results_dir, exist_ok=True)

    # Create data inverse scaler
    inverse_scaler = datasets.get_data_inverse_scaler(config)

    # loads diffusion model
    diffusion_model, state = load_diffusion_model(config, workdir)

    # define the diffusion model
    diffusion = GaussianDiffusion(config.model.beta_min, config.model.beta_max, T=1000)

    # sampling
    sampling_shape = (config.eval.batch_size, config.data.num_channels, config.data.image_size, config.data.image_size)

    if generate:
        # generates 10 samples at times 1, 100, 200, ,..., 1000
        samples, ts = progressive_generation(config, diffusion, diffusion_model, sampling_shape, inverse_scaler,
                                             n_samples=11)

        # save images for reproducibility
        with tf.io.gfile.GFile(
                os.path.join(results_dir, f"samples_0.npz"), "wb") as fout:
            io_buffer = io.BytesIO()
            np.savez_compressed(io_buffer, samples=samples.cpu())
            fout.write(io_buffer.getvalue())

    else:
        # load generated samples
        files = glob.glob(os.path.join(results_dir, 'samples_*.npz'))

        # # Load the npz file
        data = np.load(files[0])
        samples = torch.tensor(data["samples"])
        ts = np.linspace(0, diffusion.T - 1, 11, dtype=int)
    # save samples
    if config.data.image_size <= 64:
        plts.save_image(torch.clip(samples[-1] * 255, 0, 255).int(), results_dir, n=36, pos="square", padding=0,
                        name="{}_samples".format(config.data.dataset.lower()), show=False)
    else:
        plts.save_image(torch.clip(samples[-1] * 255, 0, 255).int(), results_dir, n=16, pos="square", padding=0,
                        name="{}_samples".format(config.data.dataset.lower()), show=False)

    # Plot the potential
    alpha = torch.linspace(-0.2 * torch.pi, 0.5 * torch.pi + 0.2 * torch.pi, 20)

    # logging.info("Generating interpolations ")
    # # Sanity check of the interpolation over ts
    # for t_idx in range(0, len(samples), 2):
    #     # interpolate over x1 and x2 over ts diffusion times
    #     x1 = samples[t_idx][0]
    #     x2 = samples[t_idx][1]
    #     t = 999 - ts[t_idx]
    #     x = torch.cos(alpha)[:, None, None, None] * x1 + torch.sin(alpha)[:, None, None, None] * x2
    #     plts.save_image(torch.clip(x * 255, 0, 255).int(), results_dir, n=len(alpha), w=3, padding=0, nrow=20, name="{}_interpolations_{}_{}".format(config.data.dataset.lower(), t_idx, t),  pos="vertical", show=False)

    # Compute the potential over interpolated trajectories for different diffusion times

    # Define parameters
    alpha = torch.linspace(-0.2 * torch.pi, 0.5 * torch.pi + 0.2 * torch.pi, grid_size)
    delta_alpha = alpha[1] - alpha[0]

    # Create subplots
    fig, axs = plt.subplots(1, 3, figsize=(15, 4), sharey=False)

    logging.info("Evaluating the potential over alphas")
    num_images = 20  # samples.shape[1]
    # Iterate over desired time steps
    m = 1
    for ax_idx, t_idx in enumerate([3, 9, 10]):
        # iterate over the batch for each pair of images (i, j) with i != j
        potentials = []
        for i in range(num_images):
            for j in range(i + 1, num_images):
                x1 = samples[t_idx][i]
                x2 = samples[t_idx][j]
                t = 999 - ts[t_idx]
                # iterpolate image over pairwise images
                x = torch.cos(alpha)[:, None, None, None] * x1 + torch.sin(alpha)[:, None, None, None] * x2
                x = x.to(config.device)

                # Calculate potential
                potential = evaluate_potential(x, x1, x2, t, diffusion_model, diffusion, alpha, delta_alpha, config)
                potentials.append(potential)
        potentials = np.stack(potentials, axis=0)
        print(potentials.shape)
        # Plot potential
        ax = axs[ax_idx]
        ax.plot(alpha, np.mean(potentials, axis=0))
        #         ax.legend()

        #         ax.set_xlabel(r'$\alpha$')
        #         if j in [0, 1]:
        #             ax.legend(loc='upper center')
        ax.axvline(x=0, ls='--', label=r"$y_1$", color="k", linewidth=1)
        ax.axvline(x=1.5, ls='--', label=r"$y_2$", color="r", linewidth=1)

        # add text centered
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2

        if t == 999:
            ax.text(center_x, center_y, r"$T- t$={}".format(t + 1), fontsize=12, ha='center', va='center')
        else:
            ax.text(center_x, center_y, r"$T- t$={}".format(t), fontsize=12, ha='center', va='center')

        if m:
            ax.set_ylabel(r'$u(\mathbf{x}, t)$')
            m = 0
        ax.legend()
        ax.set_xlabel(r'$\alpha$')

    # Adjust the spacing between subplots and save the figure
    plt.subplots_adjust(wspace=0.2)
    fig.savefig(os.path.join(results_dir, "potentials_{}_avg_{}.png".format(config.data.dataset.lower(), num_images)),
                dpi=200, format='png', bbox_inches='tight')

    # Show images
    for img_name, img in zip(
            ["{}_im1".format(config.data.dataset.lower()), "{}_im2".format(config.data.dataset.lower())], [x1, x2]):
        plts.show_image(torch.clip(img * 255, 0, 255).int(), results_dir, name=img_name, w=4, show=False)

    logging.info("Plotting the progressive generation reduced")

    C, H, W = config.data.num_channels, config.data.image_size, config.data.image_size
    # remove the first n_remove=4 samples corresponding to times 1000, 900, 800, 700,
    n_remove = 4
    n_to_show = 2
    n_samples = samples.shape[0] - n_remove

    # reshape the tensor to [12*9, 3, 64, 64]
    i = 20
    tensor = samples[n_remove:, i:i + n_to_show].transpose(0, 1)
    tensor = tensor.reshape(-1, C, H, W)

    # create a grid of images
    grid = vutils.make_grid(tensor, nrow=n_samples, padding=0)
    grid = torch.clip(grid * 255, 0, 255).int()

    # Calculate the desired size for each image
    image_size = H  # Fixed size of each image
    scale_factor = 4  # Adjust this value to increase the image size

    # Calculate the figure size based on the desired image size and scale factor
    fig_width = image_size * n_samples * scale_factor / 100  # Adjust this value as needed
    fig_height = image_size * n_samples * scale_factor / 50  # Adjust this value as needed

    # plot the grid with increased image size
    fig = plt.figure(figsize=(fig_width, fig_height))
    plt.imshow(grid.permute(1, 2, 0).numpy())

    plt.axis('off')
    name = "{}_progressive_generation".format(config.data.dataset.lower())
    fig.savefig(os.path.join(results_dir, "{}.{}".format(name, "png")), bbox_inches='tight', format="png", pad_inches=0)
