import random
import subprocess
import numpy as np

TARGETS_PATH = ""
CONFIG_PATH = "orch_config.txt"

targets = None  # TODO

num_subtargets = 2

full_orchestra = ['Bn', 'ClBb', 'Fl', 'Hn', 'Ob', 'Tbn', 'TpC', 'Va', 'Vc', 'Vn']

# if possible, this should be a list of functions where each function takes two parameters
# the first parameter is audio as a numpy array (the output of librosa.load)
# the second parameter is the number of subtargets to separate the target into
separation_functions = []

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


def orchestrate(target_path, config_file_path):
    """
    Orchestrate the audio at 'target_path' given the parameters defined in 'config_file_path'
    :param target_path: path to the .wav to be orchestrated
    :param config_file_path: path to Orchidea config text file
    :return: None
    """
    cmd = ["./orchestrate", target_path, config_file_path]
    subprocess.run(cmd, stdout=subprocess.DEVNULL)  # this suppresses output
    # TODO: will need to rename output file here so it isn't overwritten after every loop
    # could rename with 'mv name new_name' passed to os.system()


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


if __name__ == "__main__":

    for target in targets:
        # orchestrate full (non-segmented) target with Orchidea
        full_target_error = 0
        set_config_parameter(CONFIG_PATH, 'orchestra', full_orchestra)
        for onset_threshold in thresholds:
            set_config_parameter(CONFIG_PATH, 'onsets_threshold', onset_threshold)
            orchestrate(target, CONFIG_PATH)
            # TODO: load orchestrated solution and compute distance
            full_target_error += spectral_distance()
        full_target_error /= len(thresholds)  # average of distances

        # separate target into subtargets using different separator functions

        # nested list; len(subtargets) = num_subtargets and len(subtargets[i]) = # of separation functions
        # all_subtargets[i][k] is the ith subtarget separated via separation algorithm k
        all_subtargets = [[] for _ in range(num_subtargets)]
        for separator in separation_functions:
            subtargets = separator(target, num_subtargets)
            for i in range(num_subtargets):
                all_subtargets[i].append(subtargets[i])

        # orchestrate subtargets with different segmentation thresholds
        orchestrated_subtargets =
        for subtargets in all_subtargets:
            for subtarget in subtargets:


        # create all possible combinations of orchestrated subtargets


