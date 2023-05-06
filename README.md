# Symmetry breaking in generative diffusion models

This repository is the official implementation of the paper [Symmetry breaking in generative diffusion models]()
 

---

<p style="text-align:justify">
We show the occurrence of symmetry breaking in diffusion models, dividing its generative 
dynamics into two distinct phases: 1) a linear steady-state dynamics around a central fixed-point and 2) an attractor
dynamics directed towards the data manifold. The instability of a central fixed-point divides these two phases, where
the early dynamics does not significantly contribute to the final generation with resulting window of instability being
responsible for the diversity of the generated samples. Using a Gaussian late start approximation schem 
we significantly boost performance in fast samplers while at the same time improving diversity over the generated images.
</p> 

<figure>
  <img src="./imgs/ddim_samples_diversity_5_800_tc_4.png" alt="Image 1" style="width:45%; float:left;">
  <img src="./imgs/gslddim_samples_diversity_5_500_tc_4.png" alt="Image 2" style="width:45%; float:right;">
  <figcaption>DDIM 5 denosing steps .</figcaption>
</figure>






>📋  Optional: include a graphic explaining your approach/main result, bibtex entry, link to demos, blog posts and tutorials

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

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