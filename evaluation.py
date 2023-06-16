import os
import numpy as np
import tensorflow as tf


def load_dataset_stats(config, path="./"):
    image_size = config.data.image_size
    """Load the pre-computed dataset statistics."""
    if config.data.dataset == 'MNIST':
        filename = os.path.join(path, 'assets/stats/mnist_{}_stats.npz'.format(image_size))
    elif config.data.dataset == 'CIFAR10':
        filename = os.path.join(path, 'assets/stats/cifar10_{}_stats.npz'.format(image_size))
    elif config.data.dataset == 'CELEBA':
        filename =  os.path.join(path,'assets/stats/celeba_{}_stats.npz'.format(image_size))
    elif config.data.dataset == 'CELEBA64':
        filename =  os.path.join(path,'assets/stats/celeba64_{}_stats.npz'.format(image_size))
    elif config.data.dataset == 'CelebAHQ':
        filename =  os.path.join(path,'assets/stats/celebahq_{}_stats.npz'.format(image_size))
    elif config.data.dataset == 'IMAGENET64':
        filename =  os.path.join(path,'assets/stats/imagenet64_{}_stats.npz'.format(image_size))
    elif config.data.dataset == 'LSUN':
        filename = f'./assets/stats/lsun_{config.data.category}_{config.data.image_size}_stats.npz'
    else:
        raise ValueError(f'Dataset {config.data.dataset} stats not found.')

    with tf.io.gfile.GFile(filename, 'rb') as fin:
        stats = np.load(fin)
        return stats
