import os
import io
import logging
import numpy as np
import torch
import datasets
import tensorflow as tf
from scipy import linalg
from inception import InceptionV3


def get_batch_activations(batch, model):
    """
    Calculates the activations of the pool_3 layer for a provided batch
    Args:
        batch: The input batch for which activations are calculated
        model: The model used for the activation calculation.
    Returns:
        A numPy array of shape Nx2048, where N is the size of batch.
    """
    return model(batch)[0].squeeze(3).squeeze(2).cpu().numpy()


def get_batch_stats(batch, model, device):
    """ Evaluate stats for a batch.
    Args:
        batch: tensor of shape NxCxHxW
        model: the inception model
    Returns:
        mu :  The sample mean over activations for a given batch.
        cov : The covariance matrix over activations for a given batch.
    """
    # Check the number of channels in the input batch.
    if batch.shape[1] == 1:
        # Inception expects three channels so we replicate the channel if necessary.
        batch = batch.repeat((1, 3, 1, 1))
    batch = batch.to(device)

    # Calculate the activations of the batch using the model.
    batch_activations = get_batch_activations(batch, model=model)

    # Obtain statistical measures (mean and covariance) of the activations.
    mu, cov = get_statistics_numpy(batch_activations)
    return mu, cov


def get_data_stats(train_ds, model, device):
    """
    Compute stats for a dataset.
    Args:
        train_ds: training dataset  Input tensor of shape BxCxHxW. Values are expected to be in range (0, 1)
        model: the inception model
        device: computation device (cpu or cuda)
    Returns:
        mu :  The sample mean over activations for the entire dataset.
        cov : The covariance matrix over activations for the entire dataset.
    """
    activations = np.array([])
    with torch.no_grad():
        # for batch_id in range(len(train_ds)):
        for batch_id, batch_dict in enumerate(train_ds):
            logging.info("Processing batch number: %d", batch_id)
            batch = torch.from_numpy(batch_dict['image']._numpy()).to(device).float()
            batch = batch.permute(0, 3, 1, 2)

            # Inception expects three channels so we replicate the channel if necessary.
            if batch.shape[1] == 1:
                batch = batch.repeat((1, 3, 1, 1))

            batch = batch.to(device)

            if batch_id % 20 == 0:
                logging.info("Making FID stats -- step: %d" % (batch_id))
            batch_activations = get_batch_activations(batch, model=model)

            # Accumulate the activations
            if activations.size == 0:
                activations = batch_activations
            else:
                activations = np.append(activations, batch_activations, axis=0)

    # Compute statistical measures (mean and covariance) of the activations.
    mu, cov = get_statistics_numpy(activations)
    return mu, cov


#### NOTE: Below adapted from
# https://github.com/mseitzer/pytorch-fid/blob/master/src/pytorch_fid/fid_score.py

def get_statistics_numpy(numpy_data):
    mu = np.mean(numpy_data, axis=0)
    cov = np.cov(numpy_data, rowvar=False)
    return mu, cov


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)


# Create FID stats by looping through the whole data
def fid_stats(config, fid_dir="assets/stats"):
    """ Create dataset statistics file containing the mu and Sigma for FID scores.
    Args:
      config: Configuration to use.
      fid_dir: The subfolder for storing fid statistics.
    """
    # Create directory to save data stats
    os.makedirs(fid_dir, exist_ok=True)

    # Build data pipeline on evaluation mode
    train_ds, eval_ds, _ = datasets.get_dataset(config, uniform_dequantization=False, evaluation=True)

    # Load Inception model
    device = config.device
    incept = InceptionV3().to(device)
    incept.eval()

    # obtain data distribution moments mu and conv
    mu, cov = get_data_stats(train_ds, incept, device)

    # Save data distribution moments mu and conv for FID scores
    filename = f'{config.data.dataset.lower()}_{config.data.image_size}_stats.npz'
    with tf.io.gfile.GFile(os.path.join(fid_dir, filename), "wb") as fout:
        io_buffer = io.BytesIO()
        np.savez_compressed(io_buffer, mu=mu, cov=cov)
        fout.write(io_buffer.getvalue())

#########################
# Minimal usage example #
#########################
'''device = 'mps'
incept = InceptionV3().to(device)
incept.eval()
testloader = datasets.get_test_dataloader('cifar', 128)
mu, cov = get_data_stats(testloader, incept, device)
mu_b, cov_b = get_batch_stats(next(iter(testloader))[0], incept, device)
fid = calculate_frechet_distance(mu_b, cov_b, mu, cov)'''
