import os
import subprocess
import itertools
from pathlib import Path
import random
from shutil import copyfile
import numpy as np
import soundfile as sf
import librosa

from utils import load_model, apply_model
import architectures.ConvTasNetUniversal.separate as TDCNNpp_separate
import architectures.Open-unmix.separate as OpenUnmix_separate
import architectures.Demucs.separate as Demucs_separate

# path to the TinySOL database
TINYSOL_PATH = "../TinySOL"
# path to the analysis file of TinySOL, ex: TinySOL.spectrum.db
DB_PATH = "../TinySOL.spectrum.db"
CONFIG_PATH = "orch_config.txt"
SAMPLING_RATE = 44100
targets = ["../csmc_mume_2020/targets/WinchesterBell.wav",
           "../csmc_mume_2020/targets/car-horn.wav"]
num_subtargets = 2
full_orchestra = ['Bn', 'ClBb', 'Fl', 'Hn', 'Ob', 'Tbn', 'TpC', 'Va', 'Vc', 'Vn']
TEMP_OUTPUT_PATH = "./TEMP"
TDCNNpp_model_path = "./trained_TDCNNpp"
TDCNNpp_nb_sub_targets = 4


def clearTemp():
    """
    Clear the temp directory containing the outputs of the different models
    """
    
    if os.path.exists(TEMP_OUTPUT_PATH):
        os.rmdir(TEMP_OUTPUT_PATH)


def separate(audio_path, model_name, num_subtargets, *args):
    """
    Separate the audio into the given
    :param audio_path: path to audio input (wav file)
    :param model_name: name of the model ('demucs' or...)
    :param num_subtargets: the number of subtargerts to estimate from the mix
    :param *args: any relevant additional argument (for example combinations
                    of sub targets to match num_subtargets)
    
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
            Demucs_separate.separate(audio_path, output_path, 'tasnet')
        elif model_name == "Demucs":
            Demucs_separate.separate(audio_path, output_path, 'demucs')
        elif model_name == "OpenUnmix":
            OpenUnmix_separate(audio_path, output_path)
        else:
            raise Exception("Model name must be one of those four : TDCNN, TDCNN++, OpenUnmix, Demucs")
    
    # Read sub targets and output them in numpy array format
    sub_targets = []
    sr = None
    
    if model_name == "TDCNN++":
        if num_subtargets != len(args[0]):
            raise Exception("For TDCNN++, it is required to specify the way to combine the sub targets to generate num_subtargets sub targets. Must be of the form [[0, 3], [1, 2]]")
        for l in args[0]:
            # Combine sub_targets generated according to the list in *args
            a = None
            for s in l:
                t,sr = librosa.load(output_path + "/sub_target{}.wav".format(s), sr=None)
                if a is None:
                    a = t
                else:
                    a += t
            sub_targets.append(a)

    if sr == None:
        raise Exception("No sample rate for output detected")
        
    return sub_targets, sr

    # model_path = Path('models').joinpath(str(num_subtargets) + '_sources')
    # model = load_model(model_path)
    #
    # return apply_model(model, audio, shifts=None, split=False, progress=False)



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
    

def generate_separation_functions(model_name, num_sub_targets):
    l = []
    if model_name == "TDCNN++":
        for perm in gen_perm_group(list(range(TDCNNpp_nb_sub_targets)), num_sub_targets):
            l.append(lambda a, n: separate(a, "TDCNN++", n, perm))
    return l


# a list of functions where each function takes two parameters
# the first parameter is audio
# the second parameter is the number of subtargets to separate the target into
separation_functions = generate_separation_functions("TDCNN++", num_subtargets)
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


if __name__ == "__main__":
    # distances for non-separated targets
    # full_target_distances[i] is the average across thresholds for distance of full target i
    full_target_distances = []
    # distances for separated targets
    # separated_target_distances[i] is a list of avg distances for each separation function for target i
    separated_target_distances = []

    copyfile('orch_config_template.txt', CONFIG_PATH)

    set_config_parameter(CONFIG_PATH, 'sound_paths', TINYSOL_PATH)
    set_config_parameter(CONFIG_PATH, 'db_files', DB_PATH)

    for target_path in targets:
        target, SAMPLING_RATE = librosa.load(target_path, sr=None)

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

        # separate target into subtargets using different separator functions
        print("Separating target into subtargets")
        # all_subtargets[i] is a list of subtargets as output by the ith separation function
        all_subtargets = []
        for separator in separation_functions:
            subtargets, sr = separator(target_path, num_subtargets)
            all_subtargets.append(subtargets)

        # orchestrate subtargets with different segmentation thresholds
        print("Orchestrating subtargets with different thresholds")
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
                for threshold in thresholds:
                    # orchestrate with threshold
                    set_config_parameter(CONFIG_PATH, 'onsets_threshold', threshold)
                    solution = orchestrate(subtarget, CONFIG_PATH)
                    orchestrated_subtargets[i][j].append(solution)

        # create all possible combinations of orchestrated subtargets and calculate distance
        print("Combining subtargets and calculating distance")
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

    print("Full Target Avg Distances:", full_target_distances)
    print("Separated Target Avg Distances:", separated_target_distances)

    # remove files created during pipeline
    for file in ['target.wav', 'segments.txt']:
        os.remove(file)
