import random
import subprocess
import itertools
import numpy as np
import soundfile as sf
import librosa

TARGETS_PATH = ""
CONFIG_PATH = "orch_config.txt"
SAMPLING_RATE = 44100

targets = None  # TODO

num_subtargets = 2

full_orchestra = ['Bn', 'ClBb', 'Fl', 'Hn', 'Ob', 'Tbn', 'TpC', 'Va', 'Vc', 'Vn']

# if possible, this should be a list of functions where each function takes two parameters
# the first parameter is audio as a numpy array (the output of librosa.load)
# the second parameter is the number of subtargets to separate the target into
separation_functions = []
num_separation_functions = len(separation_functions)

thresholds = []  # onset thresholds for dynamic orchestration



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
        config_file.write(value)
        config_file.write('\n')


def orchestrate(target, config_file_path):
    """
    Orchestrate 'target' given the parameters defined in 'config_file_path'
    :param target: sound file as numpy array, shape (len,)
    :param config_file_path: path to Orchidea config text file
    :return: orchestrated solution as numpy array, shape (len,)
    """
    sf.write('target.wav', target, samplerate=SAMPLING_RATE)
    cmd = ["./orchestrate", target, config_file_path]
    subprocess.run(cmd, stdout=subprocess.DEVNULL)  # this suppresses output
    solution = librosa.load('connection.wav', sr=None)
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
    return combinations


def combine(samples):
    """
    combine all samples in 'samples' to play simultaneously
    :param samples: list of audio signals where each element is a numpy array
    :return: single audio signal as numpy array, shape (len,) where len is the length of the longest sample in 'samples'
    """
    max_length = max(samples, key=lambda x: x.size)
    samples = [librosa.util.fix_length(y, max_length) for y in samples]
    combination = np.zeros(max_length)
    for sample in samples:
        combination += sample
    combination /= len(samples)  # normalize by number of samples
    return combination


if __name__ == "__main__":

    full_target_distances = []  # distances for non-separated targets
    separated_target_distances = []  # distances for separated targets

    for target in targets:
        # orchestrate full (non-separated) target with Orchidea
        full_target_distance = 0
        set_config_parameter(CONFIG_PATH, 'orchestra', full_orchestra)
        for onset_threshold in thresholds:
            set_config_parameter(CONFIG_PATH, 'onsets_threshold', onset_threshold)
            solution = orchestrate(target, CONFIG_PATH)
            full_target_distance += spectral_distance(target, solution)
        full_target_distance /= len(thresholds)  # average of distances
        full_target_distances.append(full_target_distance)

        # separate target into subtargets using different separator functions
        # all_subtargets[i] is a list of subtargets as output by the ith separation function
        all_subtargets = []
        for separator in separation_functions:
            subtargets = separator(target, num_subtargets)
            all_subtargets.append(subtargets)

        # orchestrate subtargets with different segmentation thresholds
        orchestras = assign_orchestras(num_subtargets)
        # orchestrated_subtargets[i][j][k] is
        # the jth subtarget, separated via algorithm i, orchestrated with threshold k
        orchestrated_subtargets = [[[] for _ in range(num_subtargets)] for _ in range(num_separation_functions)]
        for i in range(len(all_subtargets)):
            # for each separation algorithm
            subtargets = all_subtargets[i]
            for j in range(len(subtargets)):
                # for each subtarget
                subtarget = subtargets[j]
                orchestra = orchestras[j]
                set_config_parameter(CONFIG_PATH, 'orchestra', orchestra)
                for threshold in threshold:
                    # orchestrate with threshold
                    set_config_parameter(CONFIG_PATH, 'onsets_threshold', threshold)
                    solution = orchestrate(subtarget, CONFIG_PATH)
                    orchestrated_subtargets[i][j].append(solution)

        # create all possible combinations of orchestrated subtargets and calculate distance
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
