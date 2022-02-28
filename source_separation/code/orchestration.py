from configparser import ConfigParser
import os
import random
import subprocess
from shutil import copyfile

import librosa
import soundfile as sf

from utils import set_config_parameter
# from pipeline import SAMPLING_RATE, CONFIG_PATH

# Get configuration
config = ConfigParser(inline_comment_prefixes="#")
config.read("config.ini")


def orchestrate(target, sample_rate, config_path):
    """
    Orchestrate 'target' using Orchidea
    :param target: sound file as numpy array, shape (len,)
    :return: orchestrated solution as numpy array, shape (len,)
    """
    sf.write('target.wav', target, sample_rate)
    cmd = ["./orchestrate", "target.wav", config_path]
    subprocess.run(cmd, stdout=subprocess.DEVNULL)  # this suppresses output
    solution, _ = librosa.load('connection.wav', sr=sample_rate)
    return solution


def assign_orchestras(num_orchestras, full_orchestra):
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


def orchestrate_with_threshold(target, save_path, thresholds,
                               config_path, sample_rate):
    """
    Orchestrates 'target' with multiple thresholds, stores them on disk, and returns them
    If the orchestrations already exist on desk, they are not redone but simply loaded and returned
    :param target: audio
    :param save_path: Path object to save the orchestrations and check for existing orchestrations
    :return: list of solutions
    """
    solutions = []
    save_path.mkdir(parents=True, exist_ok=True)
    for onset_threshold in thresholds:
        save_name = "threshold" + str(onset_threshold)
        save_name = save_name.replace(".", "_") + ".wav"
        save_path = save_path / save_name
        if os.path.exists(save_path):
            # print("Found orchestration on disk")
            solution, _ = librosa.load(save_path, sr=SAMPLING_RATE)
        else:
            set_config_parameter(CONFIG_PATH, 'onsets_threshold', onset_threshold)
            solution = orchestrate(target)
            sf.write(save_path, solution, samplerate=SAMPLING_RATE)
        save_path = save_path.parent
        solutions.append(solution)
    return solutions


def clean_orch_config(config_path, sol_path, analysis_db_path):
    copyfile('orch_config_template.txt', config_path)
    set_config_parameter(config_path, 'sound_paths', sol_path)
    set_config_parameter(config_path, 'db_files', analysis_db_path)
