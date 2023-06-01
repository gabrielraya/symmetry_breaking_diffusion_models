# Symmetry Breaking in Generative Diffusion Models

This repository is the official implementation of the paper [Spontaneous Symmetry Breaking in Generative Diffusion Models]()


---

<p align="justify">
We show that the dynamics of diffusion models exhibit a "spontaneous symmetry breaking" phenomenon, dividing the generative dynamics
in two distinct phases: 1) A linear steady-state dynamics around a central fixed-point and 2) an attractor dynamics directed
towards the data manifold. These two "phases" are separated by the change in stability of the central fixed-point, with the
resulting window of instability being responsible for the diversity of the generated samples. In an intuitive sense, the dynamics of a generated sample passes from a phase of equal potentiality,
where any (synthetic) datum could be generated, to a denoising phase where the (randomly) "selected" datum is fully denoised.
An overview of spontaneous symmetry breaking in diffusion models is illustrated in the figure below:
</p> 

<p float="center">
  <img src="./imgs/main_image_hq.png" alt="Image 1" style="width:100%">
</p>

<p align="justify">
Our findings challenge the current dominant conception that the generative process of diffusion models is
essentially comprised of a single denoising phase. In particular, we show that
an accurate simulation of the early dynamics does not significantly contribute to the final generation, since early fluctuations
are reverted to the central fixed point. To leverage this insight, we propose a Gaussian late initialization scheme, which 
significantly improves model performance, achieving up to 3x FID improvements on fast samplers, while also increasing sample
diversity, e.g. racial composition of generated CelebA images (samples below for 5 denoising steps), as illustrated below: 
</p> 

<p float="center">
  <img src="./imgs/ddim_samples_diversity_5_800_tc_4.png" alt="Image 1" style="width:40%">
  <img src="./imgs/gslddim_samples_diversity_5_500_tc_4.png" alt="Image 2" style="width:40%">
  <img src="./imgs/diversity_plot.png" alt="Image 3" style="width:18%">
</p>

Our work offers a new way to understand the generative dynamics of diffusion models that has the potential to bring 
about higher performance and less biased fast-samplers.

## Demostrations and tutorials

To help you begin understanding the symmetry breaking phenomenon, we present the following 1D example.
 
|                                                            Link                                                             | Description                                                                    |
|:---------------------------------------------------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------|
|         [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](./notebooks/1d_example.ipynb)          | Symmetry Breaking in 1D diffusion model                                        |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](./notebooks/gaussian_late_initalization.ipynb) | Load our pretrained checkpoints and play with our Gaussian late initialization |
 


## Requirements

### Dependencies 

To install requirements:

```setup
pip install -r requirements.txt
```

Make sure to download the following files 

| File        | location      | Description                                                                      |
|-------------|---------------|----------------------------------------------------------------------------------|
| Stats files | `assets/stats` | Files contating the statistics (mean, and covariance) for evaluating FID scores. |
| Checkpoints | ``              | Trained diffusion model checkpoint.                                              |

 

This code implementation is partly based on the DDPM implementation by  [Song et al. (2021)](https://github.com/yang-song/score_sde_pytorch).
