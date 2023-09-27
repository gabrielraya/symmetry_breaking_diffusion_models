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
This phenomenon is clearly observable in the one-dimensional case. A symmetry-breaking event, marked by a significant change in the potential well's shape, occurs upon reaching a critical value. This triggers a split in the potential well, signifying a shift in the dynamics and effectively illustrating these two generative phases.
</p>
<p align="center">
  <img src="./imgs/one-dimensional.gif" alt="Image 1" width="75%" />
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
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ONQgBFye2MF9YSHszNvYApzqKYmh_JpA?usp=sharing) | Load our pretrained checkpoints and play with our Gaussian late initialization |
 


## Requirements

### Dependencies 

To install requirements:

```setup
pip install -r requirements.txt
```

You can directly train and evaluate the model on your particular data following the commands below. However, 
for merely reproducing the results on the paper using the pretrained models, please refer to the section [Pre-trained Models](#pre-trained-models)

### Training
To train the diffusion model on e.g., CIFAR10, run this command:

```train
python main.py --workdir="./results/ddpm_cifar10" --config="configs/ddpm/cifar10.py" --mode="train"
```

>📋  To train the model on other dataset, just replace the config file and the corresponding workdir
> For MNIST for example `python main.py --workdir="./results/ddpm_mnist" --config="configs/ddpm/mnist.py" --mode="train"`
> For CelebA 64x64 : `python main.py --workdir="./results/ddpm_celeba64" --config="configs/ddpm/celeba64.py" --mode="train"`

This will train a model using the configuration found in `configs/ddpm/XXXXX.py`. During training, the validation batch size is 
the same as the training batch size.

### Evaluation

#### Get Stats Files

To evaluate the trained model we first need to obtain the stats files, you can download the files [here]()
or by running the following command:

```eval
 python main.py --config="configs/ddpm/cifar10.py" --mode="fid_stats"
```

### FID scores

```eval
 python main.py --config="configs/ddpm/cifar10.py" --mode="fid_stats"
```


### More to be added soon here!