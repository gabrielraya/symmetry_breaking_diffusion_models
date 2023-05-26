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


def setup_logger():
    """Configure the logger to output to both console and file."""
    os.makedirs(FLAGS.workdir, exist_ok=True)
    gfile_stream = open(os.path.join(FLAGS.workdir, 'stdout.txt'), 'w')
    handler = logging.StreamHandler(gfile_stream)
    formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
    handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel('INFO')


def run_training():
    """Run the training pipeline."""
    run_lib.train(FLAGS.config, FLAGS.workdir)


def run_evaluation():
    """Run the evaluation pipeline."""
    run_lib.evaluate(FLAGS.config, FLAGS.workdir, FLAGS.eval_folder)


def generate_fid_stats():
    """Generate moment statistics mu and cov for FID scores."""
    ffid.fid_stats(FLAGS.config)


def generate_gauss_approx_stats():
    """Generate moment statistics mu and cov for Gaussian approximation."""
    fs.gauss_approx_stats(FLAGS.config)


def run_ablation_study():
    """Run ablation study for linear samplers."""
    fs.ablation(FLAGS.config, FLAGS.workdir)


def run_correlation_analysis():
    """Run correlation analysis and plot the normalized pixels."""
    run_lib.correlation_analysis(FLAGS.config, FLAGS.workdir)


def run_late_sampler_ablation():
    """Run ablation over trained model starting at different diffusion steps."""
    fs.evaluate_fids(FLAGS.config, FLAGS.workdir)


def run_given_sampler_ablation():
    """Run a given sampler."""
    fs.evaluate_fids_fast_samplers(FLAGS.config, FLAGS.workdir, FLAGS.eval_folder,
                                   FLAGS.n_steps, FLAGS.fast_sampler,
                                   gaussian_approximation=FLAGS.gaussian_approximation)


def visualize_given_sampler():
    """Run a given sampler for visualization."""
    fs.visualize_analysis(FLAGS.config, FLAGS.workdir, FLAGS.eval_folder, N=FLAGS.samples)


def visualize_potentials():
    """Visualize the potentials."""
    fs.visualize_potentials(FLAGS.config, FLAGS.workdir, FLAGS.eval_folder, FLAGS.generate, FLAGS.grid_size)


def main(argv):
    """Main function."""
    setup_logger()

    if FLAGS.mode == "train":
        run_training()
    elif FLAGS.mode == "eval":
        run_evaluation()
    elif FLAGS.mode == "fid_stats":
        generate_fid_stats()
    elif FLAGS.mode == "gauss_approx_stats":
        generate_gauss_approx_stats()
    elif FLAGS.mode == "ablation":
        run_ablation_study()
    elif FLAGS.mode == "correlation":
        run_correlation_analysis()
    elif FLAGS.mode == "late_sampler":
        run_late_sampler_ablation()
    elif FLAGS.mode == "ablation_fast_sampler":
        run_given_sampler_ablation()
    elif FLAGS.mode == "visualize_fast_sampler":
        visualize_given_sampler()
    elif FLAGS.mode == "visualize_potentials":
        visualize_potentials()
    else:
        raise ValueError(f"Mode '{FLAGS.mode}' not recognized.")


if __name__ == '__main__':
    app.run(main)
