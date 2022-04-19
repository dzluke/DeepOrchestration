import itertools
import json
import os
from pathlib import Path
from shutil import copyfile

import librosa
import numpy as np
import soundfile as sf

from orchestration import (assign_orchestras, clean_orch_config,
                           orchestrate_with_threshold)
from utils import next_power_of_2, set_config_parameter


############################################################################
# To run this file, fill in the following path variables then run the file #
############################################################################
SOL_PATH = "../OrchideaSOL2020"  # path to the SOL database
ANALYSIS_DB_PATH = "../OrchideaSOL2020.spectrum.db"  # path to the analysis file of SOL, ex: TinySOL.spectrum.db
TARGETS_PATH = "./targets"  # path to targets to be orchestrated
SAMPLE_DATABASE_PATH = "./subtargets"  # path to samples that will be combined to create targets
TASNET_UNIVERSAL_MODEL_PATH = "./trained_TDCNNpp"  # path to pretrained TDCN++ model

# If you want to save orchestrations, set the following variables
SAVED_ORCHESTRATIONS_PATH = "./saved_orchestrations"
num_orchestrations_to_save = 0

########################################################
# You shouldn't need to change the following variables #
########################################################

ALL_ORCHESTRATIONS_PATH = "/volumes/Untitled/DeepOrchestration/MLSP21/orchestrations"  # place to save every orchestration performed
# ALL_ORCHESTRATIONS_PATH = "./temp_orchestrations"
CONFIG_PATH = "orch_config.txt"  # path to the Orchidea configuration file (you shouldn't have to change this!)
SEPARATIONS_PATH = "/volumes/Untitled/DeepOrchestration/MLSP21/separated"
# SEPARATIONS_PATH = "./temp_separations"
RESULTS_PATH = "./results.json"
TARGET_METADATA_PATH = os.path.join(TARGETS_PATH, 'metadata.json')
TDCNpp_nb_subtargets = 4
SAMPLING_RATE = 44100
NUM_SUBTARGETS = 4
FULL_ORCHESTRA = ['Fl', 'Fl', 'Ob', 'Ob', 'ClBb', 'ClBb', 'Bn', 'Bn', 'Tr',
                  'Tr', 'Tbn', 'Tbn', 'Hn', 'Hn', 'Vn', 'Vn', 'Vn', 'Vn', 'Vn',
                  'Vn', 'Vn', 'Vn', 'Va', 'Va', 'Va', 'Va', 'Vc', 'Vc', 'Vc',
                  'Cb']

separation_models = ["TDCN++", "TDCN", "Demucs", "OpenUnmix", "NMF"]
# separation_models = ["TDCN++", "OpenUnmix", "NMF"]

thresholds = [1]  # onset thresholds for dynamic orchestration


# a dict of functions where each function takes two parameters
# the first parameter is audio
# the second parameter is the number of subtargets to separate the target into
separation_functions = generate_all_separation_functions(SEPARATIONS_PATH)
num_separation_functions = len(separation_functions)


#%%
def normalized_magnitude_spectrum(target, solution):
    """
    Given a target and solution as audio files, returns the normalized magnitude spectrum of each as np arrays
    The normalized magnitude spectrum is the absolute value of the real FFT, normalized by dividing by its
    largest element
    :param target: audio, shape (len,)
    :param solution: audio, shape (len,)
    :return: target normalized magnitude spectrum, solution normalized magnitude spectrum
    """
    # if the target is longer than the solution, must trim the target
    if target.size > solution.size:
        target = target[:solution.size]
        N = next_power_of_2(target.size)
    else:
        N = next_power_of_2(solution.size)

    target_spectrum = np.abs(np.fft.rfft(target, N))
    solution_spectrum = np.abs(np.fft.rfft(solution, N))

    target_max = np.max(target_spectrum) if np.max(target_spectrum) > 0 else 1
    solution_max = np.max(solution_spectrum) if np.max(solution_spectrum) > 0 else 1
    target_spectrum /= (target_max)
    solution_spectrum /= (solution_max)

    return target_spectrum, solution_spectrum


def spectral_distance(target, solution):
    """
    Calculates the spectral distance between target and solution
    :param target: audio, shape: (len,)
    :param solution: audio, shape: (len,)
    :return: scalar representing the distance
    """
    # if the target is longer than the solution, must trim the target
    if target.size > solution.size:
        target = target[:solution.size]
        N = next_power_of_2(target.size)
    else:
        N = next_power_of_2(solution.size)

    target_fft = np.abs(np.fft.rfft(target, N))
    solution_fft = np.abs(np.fft.rfft(solution, N))

    target_max = np.max(target_fft) if np.max(target_fft) > 0 else 1
    solution_max = np.max(solution_fft) if np.max(solution_fft) > 0 else 1
    target_fft /= (target_max * N)
    solution_fft /= (solution_max * N)

    lambda_1 = 0.5
    lambda_2 = 10

    sum_1 = 0
    sum_2 = 0
    for i in range(target_fft.size):
        k_target = target_fft[i]
        k_solution = solution_fft[i]
        if k_target >= k_solution:
            sum_1 += k_target - k_solution
        else:
            sum_2 += abs(k_target - k_solution)
    distance = lambda_1 * sum_1 + lambda_2 * sum_2
    return distance * 1000


def frame_distance(custom_distance_metric):
    """
    :param distance_metric: a function that takes in two parameters: target and solution
    :return: a function that will return the distance between target and solution
    """
    def custom_distance(target, solution):
        """
        Calculates a weighted sum of the distance between frames of target and
        solution, using the provided distance metric.
        :param target: audio as numpy array
        :param solution: audio as numpy array
        :return: distance as float
        """
        # target and solution should be the same length
        length = max(target.size, solution.size)
        target = librosa.util.fix_length(target, length)
        solution = librosa.util.fix_length(solution, length)

        frame_length = 4000  # 4000 samples ~ 90 ms
        target_frames = librosa.util.frame(target, frame_length, frame_length, axis=0)
        solution_frames = librosa.util.frame(solution, frame_length, frame_length, axis=0)

        distances = []
        target_energy = []
        assert target_frames.size == solution_frames.size
        for i in range(target_frames.shape[0]):
            distance = custom_distance_metric(target_frames[i], solution_frames[i])
            distances.append(distance)
            energy = np.sum(np.sqrt(np.square(target_frames[i])))
            target_energy.append(energy)
        distances = np.array(distances)
        energy = np.array(target_energy)
        weighted_sum = np.sum(distances * energy) / np.sum(energy)
        return weighted_sum.item()

    return custom_distance


def cosine_similarity(target, solution):
    """
    Calculates the cosine similarity between target and solution
    cosine_similarity(X, Y) = <X, Y> / (||X||*||Y||)
    :param target: audio, shape: (len,)
    :param solution: audio, shape: (len,)
    :return: scalar representing the similarity (distance)
    """
    target_spectrum, solution_spectrum = normalized_magnitude_spectrum(target, solution)
    dot_product = np.dot(target_spectrum, solution_spectrum)
    norms = np.linalg.norm(target_spectrum) * np.linalg.norm(solution_spectrum)
    norms = norms if norms > 0 else 1
    similarity = dot_product / norms
    similarity = 1 - similarity
    return similarity


def create_combinations(samples):
    """
    given a nested list of samples, create combinations.
    example:
        samples = [[s1, s2], [s3, s4]], where s1 through s4 are different samples
        creates combinations [s1, s3], [s1, s4], [s2, s3], [s2, s4]
        then it combines every list to be played simultaneously
        and returns a list of length 4, where the first element is s1 and s3 played together, etc
    :param samples: nested list
    :return: list of combinations
    """
    combinations = itertools.product(*samples)
    combinations = map(combine, combinations)
    return list(combinations)


def save_best_orchestration(solutions, distances, file_path):
    """
    Save the solution in 'solutions' with shortest distance in 'distances'. Writes a wav to 'file_path'
    :param solutions: list of solutions as numpy arrays
    :param distances: list of distances, where distances[i] is the distance from the target
    to solutions[i]
    :param file_path: full path to write the file to
    :return: None
    """
    index = distances.index(min(distances))
    best_solution = solutions[index]
    sf.write(file_path, best_solution, SAMPLING_RATE)
