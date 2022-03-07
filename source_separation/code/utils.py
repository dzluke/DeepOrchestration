import os

import librosa
import numpy as np


def remove_directory(path):
    for file in os.listdir(path):
        full_path = os.path.join(path, file)
        if os.path.isdir(full_path):
            remove_directory(full_path)
        else:
            os.remove(full_path)
    if os.path.isdir(path):
        os.rmdir(path)


def clear_directory(path):
    for file in os.listdir(path):
        full_path = os.path.join(path, file)
        if os.path.isdir(full_path):
            clear_directory(full_path)
        else:
            os.remove(full_path)


def mean(lst):
    return sum(lst) / len(lst)


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


def next_power_of_2(x):
    return 1 if x == 0 else 2**(x - 1).bit_length()


def combine(samples):
    """
    combine all samples in 'samples' to play simultaneously
    :param samples: list of audio signals where each element is a numpy array
    :return: single audio signal as numpy array, shape (len,) where len is the length of the longest sample in 'samples'
    """
    max_length = max(samples, key=lambda x: x.size)
    max_length = max_length.size
    samples = sorted(samples, key=lambda x: x.size, reverse=True)
    num_samples = len(samples)
    samples = [librosa.util.fix_length(y, size=max_length) for y in samples]
    combination = np.zeros(max_length)
    for sample in samples:
        combination += sample
    combination /= num_samples  # normalize by number of samples
    return combination


def combine_with_offset(metadata):
    """
    Combine samples with offsets/padding given in metadata
    :param metadata: a list of dictionaries with the following key/values:
                            audio: audio as np array
                            name: name as string
                            padding: padding as list of 2 integers
    :return: single audio signal as numpy array representing the combination of samples
    """
    samples = [np.pad(sample['audio'], sample['padding']) for sample in metadata]
    return combine(samples)
