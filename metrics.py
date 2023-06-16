import numpy as np
import torch


def compute_correlation(x_diffused):
    """
    Computes the correlation coefficient between a fixed/reference forward trajectory $q(\mathbf{x}_{1:T}|\mathbf{x}^{ref}_0)$
    and $N$ other trajectories $q(\mathbf{x}_{1:T}|\mathbf{x}^i_0)$ a given input tensor.

    Args:
        x_diffused: A tensor of shape (B, T, C, H, W) containing the input data, where
                    B is the batch size, T is the number of time steps, C is the number
                    of channels, and H and W are the height and width of the images.

    Returns:
        A numpy array of shape (B-1, T) containing the correlation coefficients
        between the fixed image at each time step in the first data set and all other
        images at different time steps in the remaining data sets for each data set in
        the batch.
    """
    # Get the number of data sets and time steps
    batch_size, num_steps = x_diffused.shape[:2]

    # Initialize the correlation matrix
    cors = np.zeros((batch_size - 1, num_steps))

    # Get the fixed image at each time step from the first data set
    ref_images = x_diffused[0]

    # Iterate over all other data sets and time steps
    for i in range(1, batch_size):
        # skips 1 because is our reference
        for j in range(num_steps):
            # reference image at time step j
            x_ref_j = torch.flatten(x_diffused[0, j].cpu()).numpy()

            # image i at diffused step j
            x_i_j = torch.flatten(x_diffused[i, j].cpu()).numpy()

            # Compute the correlation coefficient between the two images
            cor = np.corrcoef(x_ref_j, x_i_j)

            # Store the correlation coefficient in the matrix
            cors[i - 1, j] = cor[0, 1]

    return cors


def normalized_pixel_trajectories(trajectories, n_=30):
    """
    Normalizes the pixel trajectories of a tensor of shape [time_steps, num_trajectories, num_channels, height, width]
    by dividing each pixel value by the standard deviation of that pixel across all trajectories at each time step.

    Args:
        n_ : number of trajectories to plot
        trajectories (torch.Tensor): A tensor of shape [time_steps, num_trajectories, num_channels, height, width]

    Returns:
        A tensor of normalized pixel trajectories with shape [time_steps, num_trajectories]
    """

    # Choose one pixel from channel 1 at each state out of n_ trajectories
    pixel_trajectories = trajectories[:, :n_, 0, 0, 1].clone()

    # Standard deviation over batch at each time step t shape 1001xN_
    std = torch.std(pixel_trajectories, dim=1)

    # Normalize the pixel trajectories by dividing each pixel value by the standard deviation of that pixel across
    # all trajectories at each time step
    normalized_trajectories = pixel_trajectories / std[:, None]

    return normalized_trajectories
