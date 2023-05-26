import logging
import os
from absl import app, flags
from ml_collections.config_flags import config_flags

import run_lib
import fast_fid as ffid
import fast_sampler as fs

# Suppress TensorFlow warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", None, "Training configuration.")
flags.DEFINE_string("workdir", None, "Work directory.")
flags.DEFINE_enum("mode", None, ["train", "eval", "fid_stats", "ablation", "late_sampler",
                                 "ablation_fast_sampler", "visualize_fast_sampler",
                                 "visualize_potentials", "gauss_approx_stats", "correlation"],
                  "Running mode: train, eval, fid_stats, ablation, late_sampler"
                  "ablation_fast_sampler, visualize_fast_sampler, visualize_potentials, gauss_approx_stats, "
                  "or correlation")
flags.DEFINE_string("eval_folder", "eval", "The folder name for storing evaluation results")
flags.DEFINE_enum("fast_sampler", "psdm", ["ddpm", "ddim", "psdm"], "Sampling method to use")
flags.DEFINE_integer('n_steps', 10, 'Denoising steps for fast samplers.', lower_bound=0)
flags.DEFINE_integer('samples', 64, 'Number of samples to generate for visualization evaluation fast samplers.', lower_bound=1)
flags.DEFINE_boolean('gaussian_approximation', False, 'Runs Gaussian approximation.')
flags.DEFINE_boolean('generate', False, 'Generates samples.')
flags.DEFINE_integer('grid_size', 200, 'Grid size for alpha axis for visualization of the potentials.', lower_bound=1)

# Required flags
flags.mark_flags_as_required(["workdir", "config", "mode"])


def main(argv):
    if FLAGS.mode == "train":
        # Create the working directory
        os.makedirs(FLAGS.workdir, exist_ok=True)
        # Set logger so that it outputs to both console and file
        gfile_stream = open(os.path.join(FLAGS.workdir, 'stdout.txt'), 'w')
        handler = logging.StreamHandler(gfile_stream)
        formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
        handler.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler)
        logger.setLevel('INFO')
        # Run the training pipeline
        run_lib.train(FLAGS.config, FLAGS.workdir)
    elif FLAGS.mode == "eval":
        # Run the evaluation pipeline
        run_lib.evaluate(FLAGS.config, FLAGS.workdir, FLAGS.eval_folder)
    elif FLAGS.mode == "fid_stats":
        # Generate moment statistics mu and cov for fid scores
        ffid.fid_stats(FLAGS.config)
    elif FLAGS.mode == "gauss_approx_stats":
        # Generate moment statistics mu and cov for gaussian approximation
        fs.gauss_approx_stats(FLAGS.config)
    elif FLAGS.mode == "ablation":
        # Run ablation study for linear samplers
        fs.ablation(FLAGS.config, FLAGS.workdir)
    elif FLAGS.mode == "correlation":
        # Run correlation analysis and plot the normalized pixels
        run_lib.correlation_analysis(FLAGS.config, FLAGS.workdir)
    elif FLAGS.mode == "late_sampler":
        # Run ablation over trained model starting at different diffusion steps.
        fs.evaluate_fids(FLAGS.config, FLAGS.workdir)
    elif FLAGS.mode == "ablation_fast_sampler":
        # Run a given sampler
        fs.evaluate_fids_fast_samplers(FLAGS.config, FLAGS.workdir, FLAGS.eval_folder,
                                       FLAGS.n_steps, FLAGS.fast_sampler,
                                       gaussian_approximation=FLAGS.gaussian_approximation)
    elif FLAGS.mode == "visualize_fast_sampler":
        # Run a given sampler
        fs.visualize_analysis(FLAGS.config, FLAGS.workdir, FLAGS.eval_folder, N=FLAGS.samples)

    elif FLAGS.mode == "visualize_potentials":
        # fs.visualize_potentials_avg(FLAGS.config, FLAGS.workdir, FLAGS.eval_folder, FLAGS.generate, FLAGS.grid_size)
        fs.visualize_potentials(FLAGS.config, FLAGS.workdir, FLAGS.eval_folder, FLAGS.generate, FLAGS.grid_size)
    else:
        raise ValueError(f"Mode {FLAGS.mode} not recognize.")


if __name__ == '__main__':
    app.run(main)
