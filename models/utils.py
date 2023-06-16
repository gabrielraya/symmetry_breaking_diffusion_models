""""
All functions and modules related to model definition
"""

import torch


_MODELS = {}


def register_model(cls=None, *, name=None):
    """
    A decorator for registering model classes

    Args:
        cls: the Score network model class
        name: the Score network model name
    """
    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _MODELS:
            raise ValueError(f'Already registered model with name: {local_name}')
        _MODELS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def get_model(name):
    return _MODELS[name]


def create_model(config):
    """Create the noise predictor model"""
    model_name = config.model.name

    # init the corresponding score network class
    noise_model = get_model(model_name)(config)
    noise_model = noise_model.to(config.device)
    noise_model = torch.nn.DataParallel(noise_model)
    return noise_model, model_name


def get_model_fn(model, train=False):
    """
    Creates a function to give the output of the score-based model.

    Args:
        model: The score model.
        train: `True` for training and `False` for evaluation.

    Returns:
        A model function.
    """
    def model_fn(x, labels):
        """ Compute the output of the score-based model.

        Args:
            x: A mini-batch of input data.
            labels: A mini-batch of conditioning variables for time steps.
                    Should be interpreted differently for different models.

        Returns:
            A tuple of (model output, new mutable states)
        """
        if not train:
            model.eval()
            return model(x, labels)
        else:
            model.train()
            return model(x, labels)

    return model_fn


def to_flattened_numpy(x):
    """Flatten a torch tensor `x` and convert it to numpy."""
    return x.detach().cpu().numpy().reshape((-1,))


def from_flattened_numpy(x, shape):
    """Form a torch tensor with the given `shape` from a flattened numpy array `x`."""
    return torch.from_numpy(x.reshape(shape))









