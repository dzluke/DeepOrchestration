import os
import subprocess
import itertools
import random
from shutil import copyfile
import numpy as np
import soundfile as sf
import librosa
import json

import ConvTasNetUniversal.separate as TDCNNpp_separate
import open_unmix.separate as open_unmix
import demucs.separate as demucs


############################################################################
# To run this file, fill in the following path variables then run the file #
############################################################################

TINYSOL_PATH = "../TinySOL"  # path to the TinySOL database
ANALYSIS_DB_PATH = "../TinySOL.spectrum.db"  # path to the analysis file of TinySOL, ex: TinySOL.spectrum.db
TARGETS_PATH = "./database"  # path to the database of targets
TDCNNpp_model_path = "./trained_TDCNNpp"  # path to pretrained TDCNNpp model

# If you want to save orchestrations, set the following variables
SAVED_ORCHESTRATIONS_PATH = "./saved_orchestrations"
num_orchestrations_to_save = 0

########################################################
# You shouldn't need to change the following variables #
########################################################

CONFIG_PATH = "orch_config.txt"  # path to the Orchidea configuration file (you shouldn't have to change this!)
TEMP_OUTPUT_PATH = "./TEMP"
RESULTS_PATH = "./results.json"
TDCNNpp_nb_sub_targets = 4
SAMPLING_RATE = 44100
NUM_SUBTARGETS = 4
full_orchestra = ['Fl', 'Fl', 'Ob', 'Ob', 'ClBb', 'ClBb', 'Bn', 'Bn', 'Tr', 'Tr', 'Tbn', 'Tbn', 'Hn', 'Hn',
                  'Vn', 'Vn', 'Vn', 'Vn', 'Vn', 'Vn', 'Vn', 'Vn', 'Va', 'Va', 'Va', 'Va', 'Vc', 'Vc', 'Vc', 'Cb']
separation_models = ["TDCNN++", "TDCNN", "Demucs", "OpenUnmix"]
thresholds = [0.1, 0.2, 0.3]  # onset thresholds for dynamic orchestration


def clear_temp():
    """
    Clear the temp directory containing the outputs of the different models
    """
    if os.path.exists(TEMP_OUTPUT_PATH):
        remove_directory(TEMP_OUTPUT_PATH)


def remove_directory(path):
    for file in os.listdir(path):
        full_path = os.path.join(path, file)
        if os.path.isdir(full_path):
            remove_directory(full_path)
        else:
            os.remove(full_path)
    if os.path.isdir(path):
        os.rmdir(path)


def separate(audio_path, model_name, num_subtargets, *args):
    """
    Separate the audio into the given
    :param audio_path: path to audio input (wav file)
    :param model_name: name of the model ('demucs' or...)
    :param num_subtargets: the number of subtargerts to estimate from the mix
    :param *args: any relevant additional argument (for example combinations
                    of sub targets to match NUM_SUBTARGETS)

    Returns array containing sub_targets as numpy arrays in float32, and the sample rate of the output
    """

    file_name = audio_path.split("/")[-1].split(".")[0]
    output_path = TEMP_OUTPUT_PATH + "/" + model_name + "/" + file_name

    if not os.path.exists(output_path):
        if model_name == "TDCNN++":
            # ConvTasNet for Universal Sound Separation
            TDCNNpp_separate.separate(TDCNNpp_model_path + "/baseline_model",
                                                       TDCNNpp_model_path + "/baseline_inference.meta",
                                                       audio_path,
                                                       output_path)
        elif model_name == "TDCNN":
            demucs.separate(audio_path, output_path, 'tasnet')
        elif model_name == "Demucs":
            demucs.separate(audio_path, output_path, 'demucs')
        elif model_name == "OpenUnmix":
            open_unmix.separate(audio_path, output_path)
        else:
            raise Exception("Model name must be one of those four : TDCNN, TDCNN++, OpenUnmix, Demucs")

    # Read sub targets and output them in numpy array format
    sub_targets = []
    sr = None

    if model_name in separation_models:
        if num_subtargets != len(args[0]):
            raise Exception("For {}, it is required to specify the way to combine the sub targets to generate NUM_SUBTARGETS sub targets. Must be of the form [[0, 3], [1, 2]]".format(model_name))
        for l in args[0]:
            # Combine sub_targets generated according to the list in *args
            a = None
            for s in l:
                t,sr = librosa.load(output_path + "/{}.wav".format(s), sr=None)
                if a is None:
                    a = t
                else:
                    a += t
            sub_targets.append(a)
    else:
        raise Exception("Unknown model name")

    if sr == None:
        raise Exception("No sample rate for output detected")

    return sub_targets, sr

def gen_perm_group(l, n):
    if n > 0 and len(l) == 0:
        return None
    elif n == 1:
        return [[l]]
    t = gen_perm_group(l[1:], n-1)
    r = []
    if t != None:
        r = [[[l[0]]] + x for x in t]
    for i in range(1,len(l)):
        for c in itertools.combinations(l[1:], i):
            t = gen_perm_group([x for x in l[1:] if x not in c], n-1)
            if t != None:
                r.extend([[[l[0]] + list(c)] + x for x in t])
    return r


def generate_separation_function(model_name, num_sub_targets):
    l = []
    if model_name == "TDCNN++":
        init_list = ["sub_target{}".format(x) for x in range(TDCNNpp_nb_sub_targets)]
    elif model_name == "TDCNN":
        init_list = ["drums", "bass", "other", "vocals"]
    elif model_name == "Demucs":
        init_list = ["drums", "bass", "other", "vocals"]
    elif model_name == "OpenUnmix":
        init_list = ["drums", "bass", "other", "vocals"]
    else:
        raise Exception("Model name must be one of those four : TDCNN, TDCNN++, OpenUnmix, Demucs")

    for perm in gen_perm_group(init_list, num_sub_targets):
        l.append(lambda a, n: separate(a, model_name, n, perm))
    return l


def generate_all_separation_functions():
    functions = {}
    for model in separation_models:
        functions[model] = generate_separation_function(model, NUM_SUBTARGETS)[0]
    return functions


# a dict of functions where each function takes two parameters
# the first parameter is audio
# the second parameter is the number of subtargets to separate the target into
separation_functions = generate_all_separation_functions()
num_separation_functions = len(separation_functions)


def set_config_parameter(config_file_path, parameter, value):
    """
    Set the given 'parameter' to 'value' in the config file at 'config_file_path'
    The config file *must* end in a newline
    :param config_file_path: path to the Orchidea config file
    :param parameter: the parameter to be set
    :param value: the value to set the parameter to
    :return: None
    """
    with open(config_file_path, 'a') as config_file:  # 'a' means append mode
        config_file.write(parameter + " ")
        if parameter == 'orchestra':
            value = " ".join(value)  # turns a list of instruments into a string of instruments
        config_file.write(str(value))
        config_file.write('\n')


def orchestrate(target, config_file_path):
    """
    Orchestrate 'target' given the parameters defined in 'config_file_path'
    :param target: sound file as numpy array, shape (len,)
    :param config_file_path: path to Orchidea config text file
    :return: orchestrated solution as numpy array, shape (len,)
    """
    sf.write('target.wav', target, samplerate=SAMPLING_RATE)
    cmd = ["./orchestrate", "target.wav", config_file_path]
    subprocess.run(cmd, stdout=subprocess.DEVNULL)  # this suppresses output
    solution, _ = librosa.load('connection.wav', sr=None)
    return solution


def assign_orchestras(num_orchestras):
    """
    Randomly divide the full orchestra into 'num_orchestras' sub-orchestras
    :param num_orchestras: the number of orchestras you want the full orchestra divided into
    :return orchestras: a nested list of orchestras where len(orchestras) == num_orchestras
    """
    orchestras = []  # nested list of orchestras, orchestras[i] is the orchestra for segment i
    available_instruments = full_orchestra.copy()
    orchestra_size = len(available_instruments) // num_orchestras
    extra_instruments = len(available_instruments) % num_orchestras

    random.shuffle(available_instruments)

    for i in range(num_orchestras):
        orchestra = available_instruments[:orchestra_size]
        for instrument in orchestra:
            available_instruments.remove(instrument)
        orchestras.append(orchestra)

    for i in random.sample(range(num_orchestras), extra_instruments):
        instrument = available_instruments.pop()
        orchestras[i].append(instrument)

    return orchestras


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

    target_fft /= (np.max(target_fft) * N)
    solution_fft /= (np.max(solution_fft) * N)

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


def next_power_of_2(x):
    return 1 if x == 0 else 2**(x - 1).bit_length()


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


def combine(samples):
    """
    combine all samples in 'samples' to play simultaneously
    :param samples: list of audio signals where each element is a numpy array
    :return: single audio signal as numpy array, shape (len,) where len is the length of the longest sample in 'samples'
    """
    max_length = max(samples, key=lambda x: x.size)
    max_length = max_length.size
    samples = [librosa.util.fix_length(y, max_length) for y in samples]
    combination = np.zeros(max_length)
    for sample in samples:
        combination += sample
    combination /= len(samples)  # normalize by number of samples
    return combination


def mean(lst):
    return sum(lst) / len(lst)


def get_fuss_subset(num_sources):
    """
    :param num_sources: Only targets made with 'num_sources' sources will be returned
    :return: A list of paths to a subset of the FUSS database where each target
    has exactly NUM_SUBTARGETS sources
    """
    subset = []  # each element is a path to a FUSS database sample
    for file in os.listdir(TARGETS_PATH):
        directory = os.path.join(TARGETS_PATH, file)
        if os.path.isdir(directory):
            dir_len = len([_ for _ in os.listdir(directory)])
            if dir_len == num_sources:
                file_name = file.split("_")[0] + ".wav"
                full_path = os.path.join(TARGETS_PATH, file_name)
                subset.append(full_path)
    subset.sort()  # to keep consistency across different computers
    return subset

def save_best_orchestration(solutions, distances, file_path):
    """
    Save the solution in 'solutions' with shortest distance in 'distances'. Writes a wav to 'file_path'
    :param solutions: list of solutions as numpy arrays
    :param distances: list of distances, where distances[i] is the distance from the target
    to solutions[i]
    :param filename: full path to write the file to
    :return: None
    """
    index = distances.index(min(distances))
    best_solution = solutions[index]
    sf.write(file_path, best_solution, SAMPLING_RATE)


if __name__ == "__main__":
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
 
    # targets = librosa.util.find_files(TARGETS_PATH, recurse=False)
    targets = get_fuss_subset(NUM_SUBTARGETS)
    print("Database contains {} targets".format(len(targets)))
    copyfile('orch_config_template.txt', CONFIG_PATH)

    set_config_parameter(CONFIG_PATH, 'sound_paths', TINYSOL_PATH)
    set_config_parameter(CONFIG_PATH, 'db_files', ANALYSIS_DB_PATH)

    if num_orchestrations_to_save > 0:
        if not os.path.exists(SAVED_ORCHESTRATIONS_PATH):
            os.mkdir(SAVED_ORCHESTRATIONS_PATH)

    while num_completed < len(targets):
        target_path = targets[num_completed]
        target_name = target_path.split('/')[-1]
        print("Target:", target_name)
        target, sr = librosa.load(target_path, sr=None)

        # orchestrate full (non-separated) target with Orchidea
        print("Orchestrating full target")
        full_target_distance = 0
        set_config_parameter(CONFIG_PATH, 'orchestra', full_orchestra)
        full_target_solutions = []
        distances = []
        for onset_threshold in thresholds:
            set_config_parameter(CONFIG_PATH, 'onsets_threshold', onset_threshold)
            solution = orchestrate(target, CONFIG_PATH)
            full_target_solutions.append(solution)
            distance = spectral_distance(target, solution)
            distances.append(distance)
        full_target_distances.append(distances)

        # separate target into subtargets using different separator functions
        print("Separating target into subtargets")
        # all_subtargets maps a model name to a list of subtargets
        all_subtargets = {}
        for model, separator in separation_functions.items():
            subtargets, sr = separator(target_path, NUM_SUBTARGETS)
            all_subtargets[model] = subtargets

        # orchestrate subtargets with different segmentation thresholds
        print("Orchestrating subtargets with different thresholds")
        orchestras = assign_orchestras(NUM_SUBTARGETS)
        # orchestrated_subtargets[model][j][k] is
        # the jth subtarget, separated via 'model', orchestrated with threshold k
        orchestrated_subtargets = {}
        for model in separation_functions.keys():
            orchestrated_subtargets[model] = [[] for _ in range(NUM_SUBTARGETS)]

        for model, subtargets in all_subtargets.items():
            # for each separation algorithm
            for j in range(len(subtargets)):
                # for each subtarget
                subtarget = subtargets[j]
                orchestra = orchestras[j]
                set_config_parameter(CONFIG_PATH, 'orchestra', orchestra)
                for threshold in thresholds:
                    # orchestrate with threshold
                    set_config_parameter(CONFIG_PATH, 'onsets_threshold', threshold)
                    solution = orchestrate(subtarget, CONFIG_PATH)
                    orchestrated_subtargets[model][j].append(solution)

        # create all possible combinations of orchestrated subtargets and calculate distance
        # print("Combining subtargets and calculating distance")
        combinations = {}
        for model, subtargets in orchestrated_subtargets.items():
            # for each separation algorithm
            combinations[model] = create_combinations(subtargets)
            distances = []
            for solution in combinations[model]:
                distance = spectral_distance(target, solution)
                distances.append(distance)
            separated_target_distances[model].append(distances)

        # calculate ground truth
        print("Orchestrating ground truth")
        parent, name = os.path.split(target_path)
        name = name[:-4]  # remove '.wav' from the name
        sources_folder = os.path.join(parent, name + '_sources')
        sources = []
        for file in os.listdir(sources_folder):
            file = os.path.join(sources_folder, file)
            audio, _ = librosa.load(file, sr=None)
            sources.append(audio)
        assert len(sources) == NUM_SUBTARGETS
        solutions = [[] for _ in range(len(sources))]
        for i in range(len(sources)):
            source = sources[i]
            orchestra = orchestras[i]
            set_config_parameter(CONFIG_PATH, 'orchestra', orchestra)
            for threshold in thresholds:
                # orchestrate with threshold
                set_config_parameter(CONFIG_PATH, 'onsets_threshold', threshold)
                solution = orchestrate(source, CONFIG_PATH)
                solutions[i].append(solution)

        ground_truth_combinations = create_combinations(solutions)
        distances = []
        for solution in ground_truth_combinations:
            distance = spectral_distance(target, solution)
            distances.append(distance)
        ground_truth_distances.append(distances)

        if num_orchestrations_to_save > 0:
            orch_folder_path = target_name[:-4] + "_orchestrations"
            orch_folder_path = os.path.join(SAVED_ORCHESTRATIONS_PATH, orch_folder_path)
            if not os.path.exists(orch_folder_path):
                os.mkdir(orch_folder_path)
            # save target
            copyfile(target_path, os.path.join(orch_folder_path, target_name))
            # save best full target orchestration
            distances = full_target_distances[-1]
            save_best_orchestration(full_target_solutions, distances, name)
            # save best separated orchestration
            for model, combinations in combinations.items():
                distances = separated_target_distances[model][-1]
                name = os.path.join(orch_folder_path, model + "_orchestration.wav")
                save_best_orchestration(combinations, distances, name)
            # save best ground truth orchestration
            distances = ground_truth_distances[-1]
            name = os.path.join(orch_folder_path, "ground_truth_orchestration.wav")
            save_best_orchestration(ground_truth_combinations, distances, name)
            num_orchestrations_to_save -= 1

        num_completed += 1
        # save results to json
        results['num_completed'] = num_completed
        results['full_target_distances'] = full_target_distances
        results['separated_target_distances'] = separated_target_distances
        results['ground_truth_distances'] = ground_truth_distances
        with open(RESULTS_PATH, 'w') as f:
            json.dump(results, f)

        # remove temp files created during separation
        for model_name in separation_models:
            path = os.path.join(TEMP_OUTPUT_PATH, model_name, target_name[:-4])
            if os.path.exists(path):
                remove_directory(path)

    # remove files created during pipeline
    for file in ['target.wav', 'segments.txt']:
        if os.path.exists(file):
            os.remove(file)
