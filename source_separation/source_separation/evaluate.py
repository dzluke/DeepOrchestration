# This script is used to evaluate the quality of the source separation stage.
import json
import os
from argparse import ArgumentParser
from configparser import ConfigParser
from itertools import product, permutations
from pathlib import Path

import librosa
import mir_eval
import sklearn.metrics
import numpy as np
from tqdm import tqdm

from input_output import Target, load_audio_files
from tdcn.evaluate_lib import compute_metrics
from utils import frame_distance, cosine_distance

# Get configuration
config = ConfigParser(inline_comment_prefixes="#")
config.read("config.ini")
RESULTS_FNAME = "mir_eval_separation_evaluation.json"


def distance_with_permutation(distance_metric, source_waveforms, separated_waveforms):
    n_waveforms = len(source_waveforms)
    combinations = list(
        product([i for i in range(n_waveforms)], [i for i in range(n_waveforms)])
    )
    distance_matrix = np.zeros((n_waveforms, n_waveforms))
    for c in combinations:
        distance_matrix[c[0], c[1]] = distance_metric(
            source_waveforms[0], separated_waveforms[1]
        )
    candidate_permutations = np.array(list(permutations(list(range(n_waveforms)))))
    results = np.zeros((len(candidate_permutations)))
    for i in range(len(candidate_permutations)):
        results[i] = np.sum(
            [
                distance_matrix[j, candidate_permutations[i, j]]
                for j in range(len(separated_waveforms))
            ]
        )

    return np.min(results) / n_waveforms


def eval_target(ds_path, separation_models, sample_rate, metadata, target):
    # WORKAROUND (remove eventually)
    # this is a workaround because when the separations were performed, the
    # target names were incorrectly concatenated instead of being joined with a
    # "*", they were joined with a "_"
    file_struct = Target(ds_path, target.replace("*", "_"), metadata[target])
    source_waveforms = load_audio_files(
        file_struct.get_samples_paths(),
        sample_rate,
        normalize=True,
        paddings=file_struct.get_padding(),
    )
    source_waveforms = np.array(source_waveforms)

    # Evaluate for each separation model
    results = {}
    for model in separation_models:
        separated_waveforms = load_audio_files(
            file_struct.get_separated_paths(model), sample_rate, normalize=True
        )
        separated_waveforms = np.array(separated_waveforms)

        # Separation evaluation

        # GOOGLE SOUND SEPARATION EVALUATION
        # mixture_waveform, _ = librosa.load(file_struct.get_mixture_path(),
        #                                    sample_rate)
        # snr = compute_metrics(source_waveforms, separated_waveforms,
        #                       mixture_waveform)['sisnr_separated'].numpy().mean()

        # MUSEVAL EVALUATION
        # snr, _, _, _, permutation = museval.metrics.bss_eval(
        #     source_waveforms, separated_waveforms, compute_permutation=True,
        #     window=separated_waveforms.shape[-1])
        # snr = np.mean(snr)

        # MIR_EVAL EVALUATION
        sdr, sir, sar, permutation = mir_eval.separation.bss_eval_sources(
            source_waveforms, separated_waveforms
        )

        # CUSTOM EVALUATION
        # distance_metric = frame_distance(cosine_distance)  # the distance metric to be used to evaluate solutions
        # distance = distance_with_permutation(distance_metric, source_waveforms,
        #                                      separated_waveforms)

        print("\t{}\t Avg sdr: {}".format(model, np.mean(sdr)))
        results[model] = {
            "distance": str(np.mean(sdr)),
            #   "permutation": permutation.flatten().tolist()
        }
    return results


def evaluate_separation(metadata_path, ds_path, outdir, jobs=None):
    separation_models = config["separation"]["methods"].split(", ")
    sample_rate = config["audio"].getint("sample_rate")

    Path(outdir).mkdir(parents=True, exist_ok=True)

    # Get metadata
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    # Initialize results dictionary.
    results_file = os.path.join(outdir, RESULTS_FNAME)
    if os.path.isfile(results_file):
        with open(results_file, "r") as f:
            results = json.load(f)
    else:
        results = {}

    # Evaluate each target
    for idx, target in enumerate(tqdm(metadata)):
        if target not in results:  # skip if this already exists
            print("{}/{} Evaluating results for {}".format(idx, len(metadata), target))
            results[target] = eval_target(
                ds_path, separation_models, sample_rate, metadata, target
            )
            with open(os.path.join(outdir, RESULTS_FNAME), "w") as f:
                json.dump(results, f, indent=2)

    average_results = {model: [] for model in separation_models}
    for target in results:
        for model in separation_models:
            average_results[model].append(results[target][model]["distance"])

    print("Results over all targets:")
    for model in separation_models:
        # print(f"{model}:\t {np.mean([float_info(i) for i in average_results[model]])}")
        print(model, np.mean([float(i) for i in average_results[model]]))
        print(
            f"\tStandard deviation", np.std([float(i) for i in average_results[model]])
        )
    return 0


def evaluate_orchestration():
    pass


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("metadata_file", type=str)
    parser.add_argument("ds_path", type=str)
    parser.add_argument("-o", "--outdir", type=str, default=None)
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=None,
        help="Number of jobs for multiprocessing.",
    )
    args, _ = parser.parse_known_args()
    evaluate_separation(args.metadata_file, args.ds_path, args.outdir, args.jobs)
