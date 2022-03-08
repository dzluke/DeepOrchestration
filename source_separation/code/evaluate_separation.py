# From https://github.com/google-research/sound-separation
# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Metrics for source separation."""
import itertools
import json
import os
from argparse import ArgumentParser
from configparser import ConfigParser
from multiprocessing import Manager, Process

import librosa
import numpy as np
import tensorflow.compat.v1 as tf

# Get configuration
config = ConfigParser(inline_comment_prefixes="#")
config.read("config.ini")


def calculate_signal_to_noise_ratio_from_power(signal_power, noise_power,
                                               epsilon):
    """Computes the signal to noise ratio given signal_power and noise_power.

    Args:
        signal_power: A tensor of unknown shape and arbitrary rank.
        noise_power: A tensor matching the signal tensor.
        epsilon: An optional float for numerical stability, since silences
            can lead to divide-by-zero.

    Returns:
        A tensor of size [...] with SNR computed between matching slices of the
        input signal and noise tensors.
    """
    # Pre-multiplication and change of logarithm base.
    constant = tf.cast(10.0 / tf.log(10.0), signal_power.dtype)

    return constant * tf.log(
            tf.truediv(signal_power + epsilon, noise_power + epsilon))


def calculate_signal_to_noise_ratio(signal, noise, epsilon=1e-8):
    """Computes the signal to noise ratio given signal and noise.

    Args:
        signal: A [..., samples] tensor of unknown shape and arbitrary rank.
        noise: A tensor matching the signal tensor.
        epsilon: An optional float for numerical stability, since silences
            can lead to divide-by-zero.

    Returns:
        A tensor of size [...] with SNR computed between matching slices of the
        input signal and noise tensors.
    """
    def power(x):
        return tf.reduce_mean(tf.square(x), reduction_indices=[-1])

    return calculate_signal_to_noise_ratio_from_power(
            power(signal), power(noise), epsilon)


def signal_to_noise_ratio_gain_invariant(estimate, target, epsilon=1e-8):
    """Computes the signal to noise ratio in a gain invariant manner.

    This computes SNR assuming that the signal equals the target multiplied by an
    unknown gain, and that the noise is orthogonal to the target.

    This quantity is also known as SI-SDR [1, equation 5].

    This function estimates SNR using a formula given e.g. in equation 4.38 from
    [2], which gives accurate results on a wide range of inputs, and yields a
    monotonically decreasing value when target or estimate scales toward zero.

    [1] Jonathan Le Roux, Scott Wisdom, Hakan Erdogan, John R. Hershey,
    "SDR--half-baked or well done?",ICASSP 2019,
    https://arxiv.org/abs/1811.02508.
    [2] Magnus Borga, "Learning Multidimensional Signal Processing"
    https://www.diva-portal.org/smash/get/diva2:302872/FULLTEXT01.pdf

    Args:
        estimate: An estimate of the target of size [..., samples].
        target: A ground truth tensor, matching estimate above.
        epsilon: An optional float introduced for numerical stability in the
            projections only.

    Returns:
        A tensor of size [...] with SNR computed between matching slices of the
        input signal and noise tensors.
    """
    def normalize(x):
        power = tf.reduce_sum(tf.square(x), keepdims=True, reduction_indices=[-1])
        return tf.multiply(x, tf.rsqrt(tf.maximum(power, 1e-16)))

    normalized_estimate = normalize(estimate)
    normalized_target = normalize(target)
    cosine_similarity = tf.reduce_sum(
            tf.multiply(normalized_estimate, normalized_target),
            reduction_indices=[-1])
    squared_cosine_similarity = tf.square(cosine_similarity)
    normalized_signal_power = squared_cosine_similarity
    normalized_noise_power = 1. - squared_cosine_similarity

    # Computing normalized_noise_power as the difference between very close
    # floating-point numbers is not accurate enough for this case, so when
    # normalized_signal power is close to 0., we use an alternate formula.
    # Both formulas are accurate enough at the 'seam' in float32.
    normalized_noise_power_direct = tf.reduce_sum(
        tf.square(normalized_estimate
                  - normalized_target * tf.expand_dims(cosine_similarity, -1)),
        reduction_indices=[-1])
    normalized_noise_power = tf.where(
            tf.greater_equal(normalized_noise_power, 0.01),
            normalized_noise_power,
            normalized_noise_power_direct)

    return calculate_signal_to_noise_ratio_from_power(
            normalized_signal_power, normalized_noise_power, epsilon)


def snr_with_permutations(estimates, ref_sources):
    n_sources = estimates.shape[0]  # infer number of source from input arrays
    candidate_permutations = np.array(list(
            itertools.permutations(list(range(n_sources)))))
    distance_matrix = np.zeros((n_sources, n_sources))
    for est_idx in range(n_sources):
        for ref_idx in range(n_sources):
            distance_matrix[est_idx, ref_idx] = \
                signal_to_noise_ratio_gain_invariant(estimates[est_idx],
                                                     ref_sources[ref_idx])
    results = np.zeros((len(candidate_permutations)))
    for i in range(len(candidate_permutations)):
        results[i] = np.sum([distance_matrix[j, candidate_permutations[i, j]] for j in range(n_sources)])

    best_permutation = candidate_permutations[np.argmax(results)]

    return np.max(results) / n_sources, best_permutation.tolist()


def eval_one_target(subtargets_path, separated_path, separation_models,
                    sample_rate, metadata, target):
    target_metadata = metadata[target]
    ref_sources = []
    for data in target_metadata:
        audio, _ = librosa.load(os.path.join(subtargets_path,
                                             data['name'] + '.wav'),
                                sr=sample_rate)
        audio = librosa.util.normalize(audio)
        audio = np.pad(audio, data['padding'])
        ref_sources.append(audio)
    ref_sources = np.array(ref_sources)

    # WORKAROUND (remove eventually)
    # this is a workaround because when the separations were performed, the
    # target names were incorrectly concatenated instead of being joined with a
    # "*", they were joined with a "_"
    target = target.replace('*', '_')

    # Evaluate for each separation model
    results = {}
    for model in separation_models:
        # Get estimated sources(/subtargets)
        est_path = os.path.join(separated_path,
                                model,
                                target)
        est_sources_files = librosa.util.find_files(est_path)
        assert len(est_sources_files) == len(target_metadata)

        est_sources = []
        for file in est_sources_files:
            audio, _ = librosa.load(file, sr=sample_rate)
            audio = librosa.util.normalize(audio)
            est_sources.append(audio)
        est_sources = np.array(est_sources)

        snr, permutation = snr_with_permutations(est_sources, ref_sources)
        print("\t{}\t Mean sdr: {}".format(model, snr))
        results[model] = {"snr": snr,
                          "permutation": permutation}

    return results



def main(metadata_path, subtargets_path, separated_path, outdir=None,
         jobs=None):
    separation_models = config['separation']['methods'].split(", ")
    sample_rate = config['audio'].getint('sample_rate')

    # Get metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    # Initialize results dictionary.
    if jobs:
        def process_fn(shared_dict, target, subtargets_path, separated_path,
                       separation_models, sample_rate, metadata):
            shared_dict[target] = eval_one_target(subtargets_path,
                                                  separated_path,
                                                  separation_models,
                                                  sample_rate,
                                                  metadata,
                                                  target)
        manager = Manager()
        results = manager.dict()  # this is a shared dict bewteen processese
        processes = []
        for idx, target in enumerate(metadata):
            p = Process(target=process_fn,
                        args=(results, target, subtargets_path,
                              separated_path, separation_models, sample_rate,
                              metadata)
            )
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
    else:
        results = {}
        # Evaluate each target
        for idx, target in enumerate(metadata):
            print('{}/{} Evaluating results for {}'.format(idx, len(metadata),
                                                        target))
            # Get reference sources (= padded subtargets)
            results[target] = eval_one_target(subtargets_path,
                                              separated_path,
                                              separation_models,
                                              sample_rate,
                                              metadata,
                                              target)
    results = dict(results)
    if outdir is not None:
        with open(os.path.join(outdir, "results.json"), 'w') as f:
            json.dump(results, f, indent=2)

    average_results = {model: [] for model in separation_models}
    for target in results:
        for model in separation_models:
            average_results[model].append(results[target][model]['snr'])

    for model in separation_models:
        print("Results over all targets:")
        print(f"{model}:\t {np.mean(average_results[model])}")
    return 0

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('metadata_file', type=str)
    parser.add_argument('samples_path', type=str)
    parser.add_argument('separated_path', type=str)
    parser.add_argument('-o', '--outdir', type=str, default=None)
    parser.add_argument("-j", "--jobs", type=int, default=None,
                        help="Number of jobs for multiprocessing.")
    args, _ = parser.parse_known_args()
    main(args.metadata_file, args.samples_path, args.separated_path,
         args.outdir, args.jobs)
