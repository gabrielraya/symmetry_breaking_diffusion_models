# Symmetry Breaking in Generative Diffusion Models

This repository is the official implementation of the paper [Symmetry Breaking in Generative Diffusion Models]()


---

<p align="justify">
We show that the dynamics of diffusion models exhibit a "symmetry breaking" phenomenon, dividing the generative dynamics
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

## Symmetry Breaking in 1D diffusion model 

To help you begin understanding the symmetry breaking phenomenon, we present the following 1D example.
 
|                                      Link                                      | Description                                     |
|:------------------------------------------------------------------------------:|:------------------------------------------------|
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]() | Symmetry Breaking in 1D diffusion model         |
|                                [1D eample]("./1d_example.ipynb")                                | Local File |
 


## Requirements

### Dependencies 

To install requirements:

```setup
pip install -r requirements.txt
```


### Stats files for quantitative evaluation

We provide the stats file for all our experiments. You can download [`cifar10_stats.npz`](https://drive.google.com/file/d/14UB27-Spi8VjZYKST3ZcT8YVhAluiFWI/view?usp=sharing)  and save it to `assets/stats/`. Check out [#5](https://github.com/yang-song/score_sde/pull/5) on how to compute this stats file for new datasets.




>📋  Describe how to set up the environment, e.g. pip/conda/docker commands, download datasets, etc...

## Training

To train the model(s) in the paper, run this command:

```train
python train.py --input-data <path_to_data> --alpha 10 --beta 20
```

>📋  Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters.

## Evaluation

To evaluate my model on ImageNet, run:

```eval
python eval.py --model-file mymodel.pth --benchmark imagenet
```

>📋  Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below).

## Pre-trained Models

You can download pretrained models here:

- [My awesome model](https://drive.google.com/mymodel.pth) trained on ImageNet using parameters x,y,z.

>📋  Give a link to where/how the pretrained models can be downloaded and how they were trained (if applicable).  Alternatively you can have an additional column in your results table with a link to the models.

## Results

Our model achieves the following performance on :

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it.


## Contributing

>📋  Pick a licence and describe how to contribute to your code repository. 