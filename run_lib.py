""" TRAINING AND EVALUATION FOR DIFFUSION MODELS """
import os
import io
import gc

import torch
import logging
import numpy as np
import tensorflow as tf
from torch.utils import tensorboard
from inception import InceptionV3

import losses
import datasets
import sampling
import evaluation
import plots as plts
from metrics import compute_correlation
import fast_fid as ffid
import models.utils as mutils
from utils import load_diffusion_model
from models.ema import ExponentialMovingAverage
from utils import restore_checkpoint, save_checkpoint, get_time_sequence
from diffusion_lib import GaussianDiffusion
from sampling import progressive_encoding, get_deterministic_trajectories

import wandb
from torchvision.utils import make_grid
import matplotlib.pyplot as plt


def train(config, workdir):
    """
    Runs the training pipeline.

    Args:
        config: Configuration to use.
        workdir: Working directory for checkpoints and TF summaries. If this
        contains checkpoint training will be resumed from the latest checkpoint.
    """
    # 🐝 1️⃣ Start a new run to track this script for each train model
    wandb.init(
        # Set the project where this run will be logged
        project="Training_diffusions",
        # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
        name="{}_{}_data_samples".format(config.model.name, config.data.dataset.lower()),
        # Track hyperparameters and run metadata
        config=dict(config)
    )

    # 🐝 Create a wandb Table to log images, labels and predictions to
    table = wandb.Table(columns=["image", "iter", "training loss", "eval loss"])

    # Create directories for experimental logs
    sample_dir = os.path.join(workdir, "samples")
    os.makedirs(sample_dir, exist_ok=True)

    tb_dir = os.path.join(workdir, "tensorboard")
    os.makedirs(tb_dir, exist_ok=True)
    writer = tensorboard.SummaryWriter(tb_dir)

    # Initialize model.
    noise_model, model_name = mutils.create_model(config)
    optimizer = losses.get_optimizer(config, noise_model.parameters())
    ema = ExponentialMovingAverage(noise_model.parameters(), decay=config.model.ema_rate)
    state = dict(optimizer=optimizer, model=noise_model, ema=ema, step=0)

    print("Training ", model_name)

    # Create checkpoints directory
    checkpoint_dir = os.path.join(workdir, "checkpoints")
    # Intermediate checkpoints to resume training after pre-emption in cloud environments
    checkpoint_meta_dir = os.path.join(workdir, "checkpoints-meta", "checkpoint.pth")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.dirname(checkpoint_meta_dir), exist_ok=True)

    # Resume training when intermediate checkpoints are detected
    state = restore_checkpoint(checkpoint_meta_dir, state, config.device)
    initial_step = int(state['step'])

    # Build data iterators
    train_ds, eval_ds, _ = datasets.get_dataset(config, uniform_dequantization=config.data.uniform_dequantization)
    train_iter = iter(train_ds)  # pytype: disable=wrong-arg-types
    eval_iter = iter(eval_ds)  # pytype: disable=wrong-arg-types
    # Create data normalizer and its inverse
    scaler = datasets.get_data_scaler(config)
    inverse_scaler = datasets.get_data_inverse_scaler(config)

    # Save training samples
    batch = torch.from_numpy(next(train_iter)['image']._numpy()).to(config.device).float()
    batch = batch.permute(0, 3, 1, 2)
    batch = scaler(batch)
    batch = inverse_scaler(batch)
    plts.save_image(batch, workdir, pos="vertical", name="data_samples")

    # Set uo the Forward diffusion process
    diffusion = GaussianDiffusion(config.model.beta_min, config.model.beta_max, T=1000)  # defines the diffusion process

    # Build one-step training and evaluation functions
    optimize_fn = losses.optimization_manager(config)
    train_step_fn = losses.get_step_fn(diffusion, train=True, optimize_fn=optimize_fn)
    eval_step_fn = losses.get_step_fn(diffusion, train=False, optimize_fn=optimize_fn)

    # setting sampling shape
    sampling_shape = (config.training.sampling_size, config.data.num_channels, config.data.image_size, config.data.image_size)

    # In case there are multiple hosts (e.g., TPU pods), only log to host 0
    logging.info("Starting training loop at step %d." % (initial_step, ))

    num_train_steps = config.training.n_iters

    for step in range(initial_step, num_train_steps + 1):
        # Convert data to JAX arrays and normalize them. Use ._numpy() to avoid copy
        batch = torch.from_numpy(next(train_iter)['image']._numpy()).to(config.device).float()
        batch = batch.permute(0, 3, 1, 2)
        batch = scaler(batch)
        # Execute one training step
        loss = train_step_fn(state, batch)
        if step % config.training.log_freq == 0:
            logging.info("step: %d, training_loss: %.5e" % (step, loss.item()))
            writer.add_scalar("training_loss", loss, step)

            # 🐝 Log train metrics to wandb
            metrics = {"train loss": loss.item()}
            wandb.log(metrics, step=step)

        # Save a temporary checkpoint to resume training after pre-emption periodically
        if step != 0 and step % config.training.snapshot_freq_for_preemption == 0:
            save_checkpoint(checkpoint_meta_dir, state)

        # Report the loss on an evaluation dataset periodically
        if step % config.training.eval_freq == 0:
            eval_batch = torch.from_numpy(next(eval_iter)['image']._numpy()).to(config.device).float()
            eval_batch = eval_batch.permute(0, 3, 1, 2)
            eval_batch = scaler(eval_batch)
            eval_loss = eval_step_fn(state, eval_batch)
            logging.info("step: %d, eval_loss: %.5e" % (step, eval_loss.item()))
            writer.add_scalar("eval_loss", eval_loss.item(), step)

            # 🐝 Log train and validation metrics to wandb
            val_metrics = {"eval loss": eval_loss.item()}
            wandb.log(val_metrics, step=step)


        # Save a checkpoint periodically and generate samples if needed
        if step != 0 and step % config.training.snapshot_freq == 0 or step == num_train_steps:
            # Save the checkpoint.
            save_step = step // config.training.snapshot_freq
            save_checkpoint(os.path.join(checkpoint_dir, f'checkpoint_{save_step}.pth'), state)

            # Generate and save samples
            if config.training.snapshot_sampling:
                ema.store(noise_model.parameters())
                ema.copy_to(noise_model.parameters())
                samples = sampling.sampling_fn(config, diffusion, noise_model, sampling_shape, inverse_scaler)
                ema.restore(noise_model.parameters())
                this_sample_dir = os.path.join(sample_dir, "iter_{}".format(step))
                os.makedirs(this_sample_dir, exist_ok=True)
                samples = torch.clip(samples * 255, 0, 255).int()
                torch.save(samples, os.path.join(this_sample_dir, "sample.pt"))
                plts.save_image(samples, this_sample_dir, n=config.training.sampling_size, pos="vertical", name="sample")


                # 🐝 2️⃣ Log metrics from your script to W&B

                if config.training.sampling_size < 64:
                    n=config.training.sampling_size
                else:
                    n=64
                sample_grid = make_grid(samples[:n], nrow=int(np.sqrt(n)), padding=0)
                table.add_data(wandb.Image(sample_grid.permute(1,2,0).to("cpu").numpy()), step, loss.item(), eval_loss.item())

                # close table
    wandb.log({"results_table":table}, commit=False)

    # 🐝 Close your wandb run
    wandb.finish()


def evaluate(config,
             workdir,
             eval_folder="eval"):
    """Evaluate trained models.

    Args:
        config: Configuration to use.
        workdir: Working directory for checkpoints.
        eval_folder: The subfolder for storing evaluation results.
                     Default to "eval".
    """

    # Create directory to eval_folder
    eval_dir = os.path.join(workdir, eval_folder)
    os.makedirs(eval_dir, exist_ok=True)

    # Create data inverse scaler
    inverse_scaler = datasets.get_data_inverse_scaler(config)

    # Loads diffusion model
    diffusion_model, state = load_diffusion_model(config, workdir)

    # Setup SDEs
    diffusion = GaussianDiffusion(config.model.beta_min, config.model.beta_max, T=1000)  # defines the diffusion process

    # checkpoint evaluation
    ckpt = config.eval.checkpoint

    # 🐝 1️⃣ Start a new run to track this script
    wandb.init(project= "Model_evaluation", name="{}_{}_data_samples".format(
        config.model.name, config.data.dataset.lower()), config=config)

    # 🐝 Create a wandb Table to log images, labels and predictions to
    table = wandb.Table(columns=["image", "fid_score", "batch_size", "Image Size"])


    # Generate and save samples when sampling is enabled
    if config.eval.enable_sampling:
        sampling_shape = (64,
                          config.data.num_channels,
                          config.data.image_size,
                          config.data.image_size)

        logging.info("Generating samples")
        samples = sampling.sampling_fn(config, diffusion, diffusion_model, sampling_shape, inverse_scaler, denoise=True)
        samples = torch.clip(samples * 255, 0, 255).int()
        logging.info("Saving generated samples at {}".format(eval_dir))
        plts.save_image(samples, eval_dir, n=64, pos="vertical", padding=0,
                        name="{}_{}_data_samples".format(config.model.name, config.data.dataset.lower()))
        # 🐝 2️⃣ Log metrics from your script to W&B
        sample_grid = make_grid(torch.tensor(samples)[:64], nrow=int(np.sqrt(64)), padding=0)

    # Generate samples and compute IS/FID/KID when enabled
    if config.eval.enable_fid:
        sampling_shape = (config.eval.batch_size,
                          config.data.num_channels,
                          config.data.image_size,
                          config.data.image_size)

        incept = InceptionV3().to(config.device)
        incept = incept.eval()

        activations = np.array([])

        num_sampling_rounds = config.eval.num_samples // config.eval.batch_size + 1
        for r in range(num_sampling_rounds):
            logging.info("sampling -- ckpt: %d, round: %d" % (ckpt, r))
            # Directory to save samples. Different for each host to avoid writing conflicts
            this_sample_dir = os.path.join(eval_dir, f"ckpt_{ckpt}")
            tf.io.gfile.makedirs(this_sample_dir)
            samples = sampling.sampling_fn(config, diffusion, diffusion_model, sampling_shape, inverse_scaler, denoise=True)
            samples = np.clip(samples.cpu().numpy() * 1., 0, 1).astype(np.float32)
            # samples = samples.reshape((-1, config.data.image_size, config.data.image_size, config.data.num_channels))
            # Write samples to disk
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
                samples = torch.tensor(samples).repeat((1, 3, 1, 1)).numpy()

            # get inception logits
            batch_activations = ffid.get_batch_activations(torch.tensor(samples).to(config.device), model=incept)
            activations = batch_activations if activations.size == 0 else np.append(activations, batch_activations, axis=0)

            # Force garbage collection again before returning logits
            gc.collect()
            break;
        # generative approximation distribution moments
        mu, cov = ffid.get_statistics_numpy(activations)

        # Save data distribution moments mu and conv for FID scores
        filename = f'{config.data.dataset.lower()}_{config.data.image_size}_stats.npz'
        with tf.io.gfile.GFile(os.path.join(this_sample_dir, filename), "wb") as fout:
            io_buffer = io.BytesIO()
            np.savez_compressed(io_buffer, mu=mu, cov=cov)
            fout.write(io_buffer.getvalue())

        # Load pre-computed dataset statistics.
        data_stats = evaluation.load_dataset_stats(config)
        mu_w, cov_w = data_stats['mu'], data_stats['cov']

        # Compute FID on all samples together.
        fid = ffid.calculate_frechet_distance(mu, cov, mu_w, cov_w)

        del mu, cov
        del mu_w, cov_w

        logging.info("ckpt-%d --- FID: %.6e" % (ckpt, fid))

        # 🐝 2️⃣ Log metrics from your script to W&B
        table.add_data(wandb.Image(sample_grid.permute(1,2,0).to("cpu").numpy()), fid,
                       samples.shape[0], str(samples.shape[1:]))

        # close table
        wandb.log({"results_fids":table}, commit=False)

        # 🐝 Close your wandb run
        wandb.finish()

    with tf.io.gfile.GFile(os.path.join(eval_dir, f"report_{ckpt}.npz"),"wb") as f:
        io_buffer = io.BytesIO()
        np.savez_compressed(io_buffer, fid=fid)
        f.write(io_buffer.getvalue())


def correlation_analysis(config, workdir, eval_folder="eval"):
    """Evaluate trained models.

    Args:
        config: Configuration to use.
        workdir: Working directory for checkpoints.
        eval_folder: The subfolder for storing evaluation results.
                     Default to "eval".
    """
    # Create directory to analysis folder
    eval_dir = os.path.join(workdir, eval_folder, "analysis")
    os.makedirs(eval_dir, exist_ok=True)

    # Create data normalizer and its inverse
    inverse_scaler = datasets.get_data_inverse_scaler(config)
    scaler = datasets.get_data_scaler(config)

    # loads diffusion model
    diffusion_model, _  = load_diffusion_model(config, workdir)

    # define the diffusion model
    diffusion = GaussianDiffusion(config.model.beta_min, config.model.beta_max, T=1000)

    dataset = config.data.dataset.lower()

    # get bach of data
    train_ds, eval_ds, _ = datasets.get_dataset(config, uniform_dequantization=config.data.uniform_dequantization, evaluation=False)
    n_samples_= 300
    rounds = int(np.ceil(n_samples_/config.training.batch_size))
    flag=1

    cors = []

    # run correlation coefficient at least for 100 trajectories
    for i in range(rounds):
        logging.info("Round #: %d / %d" % (i, rounds))
        # generate data for new round
        batch = torch.from_numpy(next(iter(train_ds))['image']._numpy()).to(config.device).float()
        batch = batch.permute(0, 3, 1, 2)
        x = scaler(batch)   # rescale image to [-1,1]

        # progressive generation only once
        if flag==1:
            n_samples=12
            x_diffused = progressive_encoding(x, diffusion, config, n_samples=n_samples)
            for i in range(6):
                plts.save_image(inverse_scaler(x_diffused[i]).clamp(0,1), eval_dir, n=n_samples+1, scale=1, padding=0, pos="horizontal", name="{}_forward_process_{}".format(dataset, i), show=False)
            flag=0

        # progressive generation obtaining the 1000 samples
        x_diffused = progressive_encoding(x, diffusion, config, n_samples=1000)

        # run correlation analysis over perturbed samples
        cors.append(compute_correlation(x_diffused))

        del x_diffused, x
    # stack correlation trajectories if needed
    cors = np.vstack(cors)

    logging.info("Number of trajectories: %d" % (cors.shape[0], ))

    fig = plt.figure(figsize=(7,4))
    for cor in cors[:n_samples_]:
        plt.plot(np.arange(0,len(cor)), cor)
        plt.gca().invert_xaxis()
    plt.xlabel("Diffusion time steps $t$")
    plt.ylabel("Correlation coefficient")
    fig.savefig(os.path.join(eval_dir, "{}_correlation_trajectories.png".format(dataset)), dpi=200, format="png", bbox_inches='tight')


    # deterministic generative dynamics
    shape  = (config.eval.batch_size, config.data.num_channels, config.data.image_size, config.data.image_size)
    x = diffusion.prior_sampling(shape).to(config.device)

    # Run DDIM
    sampling_fn = get_deterministic_trajectories(config, diffusion, diffusion_model)
    seq, seq_next = get_time_sequence(1000, diffusion.T)
    samples, trajectories = sampling_fn(x, seq, seq_next)
    plts.save_image(torch.clip(inverse_scaler(samples)* 255, 0, 255).int(), eval_dir, n=25, pos="vertical", padding=0,  w=4)


    n = 1001
    std = torch.std(trajectories, dim=(1))
    normalized_trajectories = (trajectories[:, 0, 0, :, :]/std[:,0, :,:]).reshape(n,config.data.image_size*config.data.image_size)

    logging.info("Normalized trajectories: %d" % (normalized_trajectories.shape[0], ))
    fig = plt.figure(figsize=(7,4))
    plt.plot(torch.flip(normalized_trajectories[:, :], dims=[0]))
    plt.xlabel("Denoising time step $t$")
    plt.ylabel("Normalized Pixels")
    fig.savefig(os.path.join(eval_dir, "{}_normalized_trajectories_flipped.png".format(dataset)), dpi=200, format="png", bbox_inches='tight')
    plt.show()
