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
DB_PATH = "../TinySOL.spectrum.db"  # path to the analysis file of TinySOL, ex: TinySOL.spectrum.db
TARGETS_PATH = "../csmc_mume_2020/targets"  # path to the database of targets
TDCNNpp_model_path = "./trained_TDCNNpp"  # path to pretrained TDCNNpp model

########################################################
# You shouldn't need to change the following variables #
########################################################

CONFIG_PATH = "orch_config.txt"  # path to the Orchidea configuration file (you shouldn't have to change this!)
TEMP_OUTPUT_PATH = "./TEMP"
RESULTS_PATH = "./results.json"
TDCNNpp_nb_sub_targets = 4
SAMPLING_RATE = 44100
NUM_SUBTARGETS = 2
full_orchestra = ['Bn', 'ClBb', 'Fl', 'Hn', 'Ob', 'Tbn', 'TpC', 'Va', 'Vc', 'Vn']


def clear_temp():
    """
    Clear the temp directory containing the outputs of the different models
    """
    if os.path.exists(TEMP_OUTPUT_PATH):
        clear_directory(TEMP_OUTPUT_PATH)


def clear_directory(path):
    for file in os.listdir(path):
        full_path = os.path.join(path, file)
        if os.path.isdir(full_path):
            clear_directory(full_path)
            os.rmdir(full_path)
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

    if model_name in ["TDCNN++", "TDCNN", "OpenUnmix", "Demucs"]:
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
    functions = []
    for model in ["TDCNN++", "TDCNN", "Demucs", "OpenUnmix"]:
        functions.extend(generate_separation_function(model, NUM_SUBTARGETS))
    return functions


# a list of functions where each function takes two parameters
# the first parameter is audio
# the second parameter is the number of subtargets to separate the target into
separation_functions = generate_all_separation_functions()
num_separation_functions = len(separation_functions)

thresholds = [0, 0.3, 0.9]  # onset thresholds for dynamic orchestration


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


if __name__ == "__main__":
    num_completed = 0
    # distances for non-separated targets
    # full_target_distances[i] is the average across thresholds for distance of full target i
    full_target_distances = []
    # distances for separated targets
    # separated_target_distances[i] is a list of avg distances for each separation function for target i
    separated_target_distances = []

    # a dictionary that represents the current state of the pipeline, including how many targets have
    # been completed, and the distances.
    # this is stored/loaded as a json so the process can be stopped and restarted
    results = {'num_completed': num_completed,
               'full_target_distances': full_target_distances,
               'separated_target_distances': separated_target_distances}

    # load saved results from json, or create json if none exists
    try:
        results_file = open(RESULTS_PATH, 'r')
        results = json.load(results_file)
        if len(results) != 0:  # if it's not an empty json
            num_completed = results['num_completed']
            full_target_distances = results['full_target_distances']
            separated_target_distances = results['separated_target_distances']
    except FileNotFoundError:
        results_file = open(RESULTS_PATH, 'w')

    results_file.close()

    targets = librosa.util.find_files(TARGETS_PATH, recurse=False)
    copyfile('orch_config_template.txt', CONFIG_PATH)

    set_config_parameter(CONFIG_PATH, 'sound_paths', TINYSOL_PATH)
    set_config_parameter(CONFIG_PATH, 'db_files', DB_PATH)

    while num_completed < len(targets):
        target_path = targets[num_completed]
        print("Target:", target_path.split('/')[-1])
        target, _ = librosa.load(target_path, sr=None)
        # orchestrate full (non-separated) target with Orchidea
        print("Orchestrating full target")
        full_target_distance = 0
        set_config_parameter(CONFIG_PATH, 'orchestra', full_orchestra)
        for onset_threshold in thresholds:
            set_config_parameter(CONFIG_PATH, 'onsets_threshold', onset_threshold)
            solution = orchestrate(target, CONFIG_PATH)
            full_target_distance += spectral_distance(target, solution)
        full_target_distance /= len(thresholds)  # average of distances
        full_target_distances.append(full_target_distance)
        print("Full target distance:", full_target_distance)

        # separate target into subtargets using different separator functions
        print("Separating target into subtargets")
        # all_subtargets[i] is a list of subtargets as output by the ith separation function
        all_subtargets = []
        for separator in separation_functions:
            subtargets, sr = separator(target_path, NUM_SUBTARGETS)
            all_subtargets.append(subtargets)

        # orchestrate subtargets with different segmentation thresholds
        print("Orchestrating subtargets with different thresholds")
        orchestras = assign_orchestras(NUM_SUBTARGETS)
        # orchestrated_subtargets[i][j][k] is
        # the jth subtarget, separated via algorithm i, orchestrated with threshold k
        orchestrated_subtargets = [[[] for _ in range(NUM_SUBTARGETS)] for _ in range(num_separation_functions)]
        for i in range(len(all_subtargets)):
            # for each separation algorithm
            subtargets = all_subtargets[i]
            for j in range(len(subtargets)):
                # for each subtarget
                subtarget = subtargets[j]
                orchestra = orchestras[j]
                set_config_parameter(CONFIG_PATH, 'orchestra', orchestra)
                for threshold in thresholds:
                    # orchestrate with threshold
                    set_config_parameter(CONFIG_PATH, 'onsets_threshold', threshold)
                    solution = orchestrate(subtarget, CONFIG_PATH)
                    orchestrated_subtargets[i][j].append(solution)

        # create all possible combinations of orchestrated subtargets and calculate distance
        # print("Combining subtargets and calculating distance")
        # distances[i] is the avg distance for separation algorithm i
        distances = []
        for subtargets in orchestrated_subtargets:
            # for each separation algorithm
            combinations = create_combinations(subtargets)
            distance = 0
            for solution in combinations:
                distance += spectral_distance(target, solution)
            distance /= len(combinations)
            distances.append(distance)
        separated_target_distances.append(distances)
        print("Separated target distance:", sum(distances) / len(distances))

        num_completed += 1
        # save results to json
        results['num_completed'] = num_completed
        results['full_target_distances'] = full_target_distances
        results['separated_target_distances'] = separated_target_distances
        with open(RESULTS_PATH, 'w') as f:
            json.dump(results, f)

    # compute average across different separation methods
    for i in range(len(separated_target_distances)):
        separated_target_distances[i] = mean(separated_target_distances[i])
        # at this point, separated_target_distances[i] is the avg distance across all methods for target i

    print("Full Target Avg Distances:", full_target_distances)
    print("Separated Target Avg Distances:", separated_target_distances)
    print("Average distance of full targets:", mean(full_target_distances))
    print("Average distance of separated targets:", mean(separated_target_distances))

    # remove files created during pipeline
    for file in ['target.wav', 'segments.txt']:
        os.remove(file)
