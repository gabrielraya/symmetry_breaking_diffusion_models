import os
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import glob
import pickle
import logging
logging.basicConfig(level = logging.INFO)
from deepface import DeepFace
import plots as plts 
import matplotlib
import seaborn as sns
matplotlib.rcParams['pdf.fonttype'] = 42
sns.set_theme(style="white")


def func(pct, allvals, absolute=False):
    if absolute: 
        absolute = int(np.round(pct/100.*np.sum(allvals)))
        return f"{pct:.1f}%\n({absolute:d})"
    else: 
        return f"{pct:.1f}%"
    

def evaluate_diversity(dataset, sampler, n_steps, tc, eval_dir, data_dir):

    # Create directory to eval_folder
    os.makedirs(eval_dir, exist_ok=True)

    # Find all files that match the format samples_<digit>.npz
    files = glob.glob(os.path.join(data_dir, 'samples_*.npz'))
    logging.info(len(files))


    results = []
    show_image = True

    # iterate over files
    for i, file in enumerate(files) :
        # Load the npz file
        data = np.load(file)
        logging.info("round {} \t file: {} \t shape= {}".format(i, file, data["samples"].shape))
        samples = torch.clip(torch.tensor(data["samples"])*255, 0,255).int()

        # Convert the tensor to BGR format
        bgr_images = torch.flip(samples.permute(0, 2,3,1), dims=[3]).numpy().astype('uint8')

        if show_image:
            plts.save_image(samples, eval_dir, n=16, pos="rectangle", nrow=8, padding=0,  name="{}_samples_diversity_{}_{}_tc".format(n_steps, sampler, tc))
            plts.save_image(samples, eval_dir, n=32, pos="rectangle", nrow=8, padding=0,  name="{}_samples_diversity_{}_{}_tc_4".format(n_steps, sampler, tc))
            show_image=False

        # for bgr_image in  tqdm(bgr_images) :
        for bgr_image in tqdm(bgr_images):
            obj = DeepFace.analyze(bgr_image, enforce_detection=False, actions=['age', 'gender', 'race', 'emotion'], silent=True)
            # Store the analysis results in a new data structure
            results.append(obj)
        # break;

    
    # Save the results to a file using the pickle module
    with open( os.path.join(eval_dir,"{}-{}-{}-diversity.pickle".format(sampler, n_steps, tc)), "wb") as f:
        pickle.dump(results, f)

    df = pd.DataFrame(columns=['Image', 'Age', 'Gender', 'Race', 'Emotion'])
    for i, obj in enumerate(results):
        age = obj[0]['age']
        gender = obj[0]['dominant_gender']
        race = obj[0]['dominant_race']
        emotion = obj[0]['dominant_emotion']
        df.loc[i] = [f"Image {i+1}", age, gender, race, emotion]

    genders = df["Gender"].copy()
    races = df["Race"].copy()
    
    # sanity check
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.hist(races)
    fig.savefig(os.path.join(eval_dir,"{}-{}-{}_{}_races.{}".format(sampler, n_steps, tc, "race", "png")), bbox_inches='tight', pad_inches=0)
    plt.close()
    
    # Plots
    # assume 'genders' and 'races' are arrays representing the gender and race of the generated images
    gender_counts = [len(genders[genders=='Man']), len(genders[genders=='Woman'])]
    genders_ = ["Man", "Woman"] 
    fig, ax = plt.subplots(figsize=(10, 5)) #, subplot_kw=dict(aspect="equal"))
    wedges, texts, autotexts = ax.pie(gender_counts, autopct=lambda pct: func(pct, gender_counts, absolute=True),
                                      textprops=dict(color="w"))

    ax.legend(wedges, genders_,
              title="Gender",
              loc="center left",
              bbox_to_anchor=(0.95, 0, 0.5, 1))

    plt.setp(autotexts, size=8, weight="bold")
    fig.savefig(os.path.join(eval_dir,"{}-{}-{}_{}.{}".format(sampler, n_steps, tc, "gender", "png")), bbox_inches='tight', dpi=300)#, pad_inches=0)
    plt.tight_layout()
    plt.close()


    # assume 'races' is an array representing the race of the generated images
    race_counts = [len(races[races=='white']), len(races[races=='latino hispanic']), len(races[races=='black']), len(races[races=='asian']), len(races[races=='middle eastern']), len(races[races=='indian'])]
    races_ = ['white', 'latino hispanic', 'black', 'asian', 'middle eastern', 'indian']
    
    # Create a pie chart with labels and percentages
    percentages = ['{:.1f}%'.format(100.0 * count / sum(race_counts)) for count in race_counts]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    fig, ax = plt.subplots(figsize=(10, 5))
    wedges, texts, autotexts = ax.pie(race_counts, autopct=lambda pct: func(pct, race_counts),
                                      textprops=dict(color="k"))
    ax.legend(wedges, races_,
              title="Race",
              loc="center left",
              bbox_to_anchor=(0.95, 0, 0.5, 1),
              fontsize= "8")


    plt.setp(autotexts, size=8) #, weight="bold")

    # ax.pie(race_counts, colors=colors, startangle=0, autopct='%1.1f%%') 
    fig.savefig(os.path.join(eval_dir,"{}-{}-{}_{}.{}".format(sampler, n_steps, tc, "race", "png")), bbox_inches='tight', dpi=300)
    plt.tight_layout() 
    plt.close()


def main():
#     dataset = "celeba64"
#     sampler = "ddim"
#     n_steps = 5
#     tc = 800
#     eval_dir= "./results/ddpm_celeba64/eval_diversity/ddim_fids_ckpt_16_n_steps_5/"
#     data_dir= "./results/ddpm_celeba64/eval_diversity/ddim_fids_ckpt_16_n_steps_5/time_t_800"

#     logging.info("Hello")
#     evaluate_diversity(dataset, sampler, n_steps, tc, eval_dir, data_dir)
    
    
#     logging.info("Running the other method")
#     dataset = "celeba64"
#     sampler = "gslddim"
#     n_steps = 5
#     tc = 500
#     eval_dir= "./results/ddpm_celeba64/eval_diversity/ddim_fids_ckpt_16_n_steps_5_gauss_approx/"
#     data_dir= "./results/ddpm_celeba64/eval_diversity/ddim_fids_ckpt_16_n_steps_5_gauss_approx/time_t_{}".format(tc)    
#     evaluate_diversity(dataset, sampler, n_steps, tc, eval_dir, data_dir)
    
    
    # logging.info("Running the other method")
    # dataset = "celeba64"
    # sampler = "data"
    # n_steps = 0
    # tc = 0
    # eval_dir= "./results/ddpm_celeba64/eval_diversity/data/"
    # data_dir= "./results/ddpm_celeba64/eval_diversity/data/training_set"    
    # evaluate_diversity(dataset, sampler, n_steps, tc, eval_dir, data_dir)

    #PSDM
#     dataset = "celeba64"
#     sampler = "psdm"
#     n_steps = 5
#     tc = 800
#     eval_dir= "./results/ddpm_celeba64/eval_diversity/psdm_fids_ckpt_16_n_steps_5/"
#     data_dir= "./results/ddpm_celeba64/eval_diversity/psdm_fids_ckpt_16_n_steps_5/time_t_800"

#     logging.info("Hello")
#     evaluate_diversity(dataset, sampler, n_steps, tc, eval_dir, data_dir)
    
    
    # logging.info("Running the other method")
    # dataset = "celeba64"
    # sampler = "gslpsdm"
    # n_steps = 5
    # tc = 600
    # eval_dir= "./results/ddpm_celeba64/eval_diversity/psdm_fids_ckpt_16_n_steps_5_gauss_approx/"
    # data_dir= "./results/ddpm_celeba64/eval_diversity/psdm_fids_ckpt_16_n_steps_5_gauss_approx/time_t_{}".format(tc)    
    # evaluate_diversity(dataset, sampler, n_steps, tc, eval_dir, data_dir)

    # logging.info("Classical DDPM-1000")
    # dataset = "celeba64"
    # sampler = "standard_sampler"
    # n_steps = 1000
    # tc = 1000
    # eval_dir= "./results/ddpm_celeba64/eval_diversity/standard/"
    # data_dir= "./results/ddpm_celeba64/eval/fids_ckpt_16_1/time_t_1000"    
    # evaluate_diversity(dataset, sampler, n_steps, tc, eval_dir, data_dir)
    
    
#     dataset = "celeba64"
#     sampler = "ddim"
#     n_steps = 10
#     tc = 900
#     eval_dir= "./results/ddpm_celeba64/eval_diversity/ddim_fids_ckpt_16_n_steps_{}/".format(n_steps)
#     data_dir= "./results/ddpm_celeba64/eval_diversity/ddim_fids_ckpt_16_n_steps_{}/time_t_{}".format(n_steps, tc)

#     logging.info("Hello")
#     evaluate_diversity(dataset, sampler, n_steps, tc, eval_dir, data_dir)
    
#     dataset = "celeba64"
#     sampler = "ddim"
#     n_steps = 3
#     tc = 666
#     eval_dir= "./results/ddpm_celeba64/eval_diversity/ddim_fids_ckpt_16_n_steps_{}/".format(n_steps)
#     data_dir= "./results/ddpm_celeba64/eval_diversity/ddim_fids_ckpt_16_n_steps_{}/time_t_{}".format(n_steps, tc)

#     logging.info("Hello")
#     evaluate_diversity(dataset, sampler, n_steps, tc, eval_dir, data_dir)

    logging.info("Running the other method")
    dataset = "celeba64"
    sampler = "gslddim"
    n_steps = 3
    tc = 400
    eval_dir= "./results/ddpm_celeba64/eval_diversity/ddim_fids_ckpt_16_n_steps_{}_gauss_approx/".format(n_steps)   
    data_dir= "./results/ddpm_celeba64/eval_diversity/ddim_fids_ckpt_16_n_steps_{}_gauss_approx/time_t_{}".format(n_steps, tc)    
    evaluate_diversity(dataset, sampler, n_steps, tc, eval_dir, data_dir)
    
    logging.info("Running the other method")
    dataset = "celeba64"
    sampler = "gslddim"
    n_steps = 10
    tc = 500
    eval_dir= "./results/ddpm_celeba64/eval_diversity/ddim_fids_ckpt_16_n_steps_{}_gauss_approx/".format(n_steps)   
    data_dir= "./results/ddpm_celeba64/eval_diversity/ddim_fids_ckpt_16_n_steps_{}_gauss_approx/time_t_{}".format(n_steps, tc)    
    evaluate_diversity(dataset, sampler, n_steps, tc, eval_dir, data_dir)
    
if __name__ == '__main__':
    main()