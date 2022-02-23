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


if __name__ == "__main__":
    distance_metric = frame_distance(spectral_distance)  # the distance metric to be used to evaluate solutions

    num_completed = 0
    # full_target_distances is a nested list of distances of orchestrating full targets without separation
    # full_target_distances[i][k] is the distance between target i and the orchestration of target i with threshold k
    full_target_distances = []
    # distances for separated targets
    # separated_target_distances['model'][i] is a list of length 81 of the distances for all possible combinations (3^4)
    # of target i after being separated by 'model', orchestrated, then combined
    separated_target_distances = {}
    for model in separation_models:
        separated_target_distances[model] = []
    # ground_truth_distances[i] is a list of length 81, it is the distances for all possible combinations of
    # orchestrations of the ground truth source subtargets
    ground_truth_distances = []

    # a dictionary that represents the current state of the pipeline, including how many targets have
    # been completed, and the distances.
    # this is stored/loaded as a json so the process can be stopped and restarted
    results = {'num_completed': num_completed,
               'full_target_distances': full_target_distances,
               'separated_target_distances': separated_target_distances,
               'ground_truth_distances': ground_truth_distances}

    # load saved results from json, or create json if none exists
    try:
        results_file = open(RESULTS_PATH, 'r')
        results = json.load(results_file)
        if len(results) != 0:  # if it's not an empty json
            num_completed = results['num_completed']
            full_target_distances = results['full_target_distances']
            separated_target_distances = results['separated_target_distances']
            ground_truth_distances = results['ground_truth_distances']
            print("Backing up results.json to results_backup.json")
            copyfile(RESULTS_PATH, 'results_backup.json')
    except FileNotFoundError:
        results_file = open(RESULTS_PATH, 'w')
    results_file.close()

    targets = librosa.util.find_files(TARGETS_PATH)
    print("Database contains {} targets".format(len(targets)))

    if num_orchestrations_to_save > 0 and not os.path.exists(SAVED_ORCHESTRATIONS_PATH):
        os.mkdir(SAVED_ORCHESTRATIONS_PATH)

    while num_completed < len(targets):
        target_path = targets[num_completed]
        target, _ = librosa.load(target_path, sr=SAMPLING_RATE)
        target_name = os.path.basename(target_path)
        target_name = os.path.splitext(target_name)[0]
        save_path = Path(ALL_ORCHESTRATIONS_PATH) / target_name

        print("Target:", target_name)
        print("num completed:", num_completed)

        clean_orch_config(CONFIG_PATH, SOL_PATH, ANALYSIS_DB_PATH)

        # orchestrate full (non-separated) target with Orchidea
        print("Orchestrating full target")
        full_target_distance = 0
        set_config_parameter(CONFIG_PATH, 'orchestra', FULL_ORCHESTRA)
        full_target_solutions = []
        distances = []
        save_path = save_path / "full_orchestration"

        solutions = orchestrate_with_threshold(target, save_path)
        full_target_solutions = solutions
        for solution in solutions:
            distance = distance_metric(target, solution)
            distances.append(distance)
        full_target_distances.append(distances)
        save_path = save_path.parent

        # separate target into subtargets using different separator functions
        print("Separating target into subtargets")
        # all_subtargets maps a model name to a list of subtargets
        all_subtargets = {}
        for model, separator in separation_functions.items():
            subtargets, sr = separator(target_path, NUM_SUBTARGETS)
            all_subtargets[model] = subtargets

        # orchestrate subtargets with different segmentation thresholds
        print("Orchestrating subtargets with different thresholds")
        orchestras = assign_orchestras(NUM_SUBTARGETS, FULL_ORCHESTRA)
        # orchestrated_subtargets[model][j][k] is
        # the jth subtarget, separated via 'model', orchestrated with threshold k
        orchestrated_subtargets = {}
        for model in separation_functions.keys():
            orchestrated_subtargets[model] = [[] for _ in range(NUM_SUBTARGETS)]

        for model, subtargets in all_subtargets.items():  # for each separation algorithm
            save_path = save_path / model
            for j in range(len(subtargets)):  # for each subtarget
                subtarget = subtargets[j]
                orchestra = orchestras[j]
                set_config_parameter(CONFIG_PATH, 'orchestra', orchestra)
                save_name = "subtarget" + str(j)
                save_path = save_path / save_name
                solutions = orchestrate_with_threshold(subtarget, save_path)
                orchestrated_subtargets[model][j] = solutions
                save_path = save_path.parent
            save_path = save_path.parent


        # create all possible combinations of orchestrated subtargets and calculate distance
        # print("Combining subtargets and calculating distance")
        combinations = {}
        for model, subtargets in orchestrated_subtargets.items():
            # for each separation algorithm
            combinations[model] = create_combinations(subtargets)
            distances = []
            for solution in combinations[model]:
                distance = distance_metric(target, solution)
                distances.append(distance)
            separated_target_distances[model].append(distances)

        # calculate ground truth
        print("Orchestrating ground truth")
        # read metadata
        with open(TARGET_METADATA_PATH, 'r') as metadata_file:
            metadata = json.load(metadata_file)
        metadata = metadata[target_name]
        sources = []
        for data in metadata:
            audio, _ = librosa.load(data['path'], sr=SAMPLING_RATE)
            audio = np.pad(audio, data['padding'])
            sources.append(audio)
        if len(sources) != NUM_SUBTARGETS:
            if len(sources) == 2:
                # this biases the results
                orchestras[0] += orchestras[2]
                orchestras[1] += orchestras[3]
                orchestras = orchestras[:2]
            else:
                raise Exception("Code is not ready to handle {} sources".format(len(sources)))
        ground_truth_solutions = [[] for _ in range(len(sources))]
        save_path = save_path / "ground_truth"
        for i in range(len(sources)):
            source = sources[i]
            orchestra = orchestras[i]
            set_config_parameter(CONFIG_PATH, 'orchestra', orchestra)
            save_name = "source" + str(i)
            save_path = save_path / save_name

            solutions = orchestrate_with_threshold(source, save_path)
            ground_truth_solutions[i] = solutions
            save_path = save_path.parent
        save_path = save_path.parent

        ground_truth_combinations = create_combinations(ground_truth_solutions)
        distances = []
        for solution in ground_truth_combinations:
            distance = distance_metric(target, solution)
            distances.append(distance)
        ground_truth_distances.append(distances)

        if num_orchestrations_to_save > 0:
            orch_folder_path = os.path.join(SAVED_ORCHESTRATIONS_PATH, target_name)
            if not os.path.exists(orch_folder_path):
                os.mkdir(orch_folder_path)
            # save target
            copyfile(target_path, os.path.join(orch_folder_path, target_name + '.wav'))
            # save best full target orchestration
            distances = full_target_distances[-1]
            name = os.path.join(orch_folder_path, "full_orchestration.wav")
            save_best_orchestration(full_target_solutions, distances, name)
            # save best separated orchestration
            for model, combos in combinations.items():
                distances = separated_target_distances[model][-1]
                name = os.path.join(orch_folder_path, model + "_orchestration.wav")
                save_best_orchestration(combos, distances, name)
            # save best ground truth orchestration
            distances = ground_truth_distances[-1]
            name = os.path.join(orch_folder_path, "ground_truth_orchestration.wav")
            save_best_orchestration(ground_truth_combinations, distances, name)
            # save distances in .txt file
            with open(os.path.join(orch_folder_path, 'distances.txt'), 'w') as f:
                f.write("full target distance: {}\n".format(full_target_distances[-1]))
                f.write("ground truth distance: {}\n".format(ground_truth_distances[-1]))
                for model, combos in combinations.items():
                    f.write("{} distance: {}\n".format(model, separated_target_distances[model][-1]))
            # move source separation results
            separations_folder = os.path.join(orch_folder_path, 'separations')
            if not os.path.exists(separations_folder):
                os.mkdir(separations_folder)
            for model_name in separation_models:
                # WORKAROUND (remove eventually)
                target_name = target_name.replace("*", "_")
                folder = os.path.join(SEPARATIONS_PATH, model_name, target_name)
                if not os.path.exists(os.path.join(separations_folder, model_name)):
                    os.mkdir(os.path.join(separations_folder, model_name))
                for source in os.listdir(folder):
                    source_path = os.path.join(folder, source)
                    copyfile(source_path, os.path.join(separations_folder, model_name, source))
            num_orchestrations_to_save -= 1

        num_completed += 1
        # save results to json
        results['num_completed'] = num_completed
        results['full_target_distances'] = full_target_distances
        results['separated_target_distances'] = separated_target_distances
        results['ground_truth_distances'] = ground_truth_distances
        with open(RESULTS_PATH, 'w') as f:
            json.dump(results, f, indent='\t')

        # remove temp files created during separation
        # for model_name in separation_models:
        #     path = os.path.join(SEPARATIONS_PATH, model_name, target_name)
        #     if os.path.exists(path):
        #         remove_directory(path)

    # remove files created during pipeline
    for file in ['target.wav', 'segments.txt']:
        if os.path.exists(file):
            os.remove(file)
