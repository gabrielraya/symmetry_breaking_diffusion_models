"""
Abstract Class for the forward and reverse process.

    It explains what we need in the diffusion process
"""
import abc
import torch


class DiffusionProcess(abc.ABC):
    """ Diffusion process abstract class. Functions are designed for a mini-batch of inputs """

    def __init__(self, T):
        """ Construct a Discrete Diffusion process.

        Args:
            N: number of time steps
        """
        super().__init__()
        self.N = T

    @property
    @abc.abstractmethod
    def T(self):
        """End time of the DP."""
        pass

    @abc.abstractmethod
    def transition_prob(self, x, t):
        pass

    @abc.abstractmethod
    def forward_step(self, x, t):
        """return sample x_t ~ q(x_t|x_{t-1})"""
        pass

    @abc.abstractmethod
    def t_step_transition_prob(self, x, t):
        """Computes the the t-step forward distribution q(x_t|x_0) """
        pass

    @abc.abstractmethod
    def t_forward_steps(self, x, t):
        """return sample x_t ~ q(x_t|x_{t-1})"""
        pass

    @abc.abstractmethod
    def prior_sampling(self, shape):
        """Generate one sample from the prior distribution, $p_T(x)$."""
        pass


class GaussianDiffusion(DiffusionProcess):
    def __init__(self, beta_min=0.1, beta_max=20, T=1000):
        """Construct a Discrete Gaussian diffusion model.

        Args:
            beta_min: value of beta(0)
            beta_max: value of beta(1)
            T: number of timesteps
        """
        super().__init__(T)
        self.beta_0 = beta_min
        self.beta_1 = beta_max
        self.N = T
        self.discrete_betas = torch.linspace(beta_min, beta_max, T)
        self.alphas = 1. - self.discrete_betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_1m_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

    @property
    def T(self):
        return self.N

    def transition_prob(self, x, timestep):
        """"Forward step as a result of the forward transition distribution q(x_t|x_t-1)
            q(x_t|x_{t-1}) = N(x_t|\sqrt{1-beta_t}x_{t-1}, \beta_t*I)
        Args:
            x : tensor NxCxHxW in range [-1,1]
            timestep: 1D tensor of size N [1,2,..., T]
        """
        beta = self.discrete_betas.to(x.device)[timestep]
        mean = torch.sqrt(1 - beta[:, None, None, None].to(x.device)) * x
        std = torch.sqrt(beta)
        return mean, std

    def forward_step(self, x, t):
        """return sample x_t ~ q(x_t|x_{t-1})
          x_t = \sqrt{1-beta_t}x_{t-1} + \sqrt{\beta_t} * z ; z~N(0,I)
        Args:
            x : tensor NxCxHxW in range [-1,1]
            t: 1D tensor of size N
        """
        mean, std = self.transition_prob(x, t)
        z = torch.randn_like(x)
        x = mean + std[:, None, None, None]*z
        return x, z

    def t_step_transition_prob(self, x, t):
        """Computes the the t-step forward distribution q(x_t|x_0)
            q(x_t|x_0) = \mathcal{N}(x_t; sqrt{\bar{\alpha_t}}x_0, (1-\bar{\alpha_t})I)
        """
        mean = self.sqrt_alphas_cumprod.to(x.device)[t, None, None, None] * x
        std  = self.sqrt_1m_alphas_cumprod.to(x.device)[t]
        return mean, std

    def t_forward_steps(self, x, t):
        """return sample x_t ~ q(x_t|x_0)
        Basically reparemeterize the t-step distribution
        x_t = sqrt{\bar{\alpha_t}}x_0 + \sqrt{(1-\bar{\alpha_t})} * z; z \sim  z~N(0,I)
        """
        mean, std = self.t_step_transition_prob(x, t)
        z = torch.randn_like(x)
        x_t = mean + std[:, None, None, None]*z
        return x_t, z

    def prior_sampling(self, shape):
        return torch.randn(*shape)
