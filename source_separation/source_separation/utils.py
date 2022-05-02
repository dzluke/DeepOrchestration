import os

import librosa
import numpy as np
import scipy.spatial


def rm_extension(fname):
    """Remove the file extension at the end of a string."""
    return os.path.splitext(fname)[0]

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
    # if target.size > solution.size:
    #     target = target[:solution.size]
    #     N = next_power_of_2(target.size)
    # else:
    #     N = next_power_of_2(solution.size)
    N = 4096
    target_spectrum = np.abs(np.fft.rfft(target, N))
    solution_spectrum = np.abs(np.fft.rfft(solution, N))

    target_max = np.max(target_spectrum) if np.max(target_spectrum) > 0 else 1
    solution_max = np.max(solution_spectrum) if np.max(solution_spectrum) > 0 else 1
    target_spectrum /= (target_max)
    solution_spectrum /= (solution_max)

    return target_spectrum, solution_spectrum


def cosine_distance(target, solution):
    """
    Calculates the cosine similarity between target and solution
    cosine_similarity(X, Y) = <X, Y> / (||X||*||Y||)
    :param target: audio, shape: (len,)
    :param solution: audio, shape: (len,)
    :return: scalar representing the similarity (distance)
    """
    target_spectrum, solution_spectrum = normalized_magnitude_spectrum(target, solution)
    # dot_product = np.dot(target_spectrum, solution_spectrum)
    # norms = np.linalg.norm(target_spectrum) * np.linalg.norm(solution_spectrum)
    # norms = norms if norms > 0 else 1
    # similarity = dot_product / norms
    # similarity = 1 - similarity
    distance = scipy.spatial.distance.cosine(target_spectrum,
                                             solution_spectrum)
    return distance


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
        Calculates a weighted sum of the distance between frames of target and solution, using the provided
        distance metric
        :param target: audio as numpy array
        :param solution: audio as numpy array
        :return: distance as float
        """
        # target and solution should be the same length
        length = max(target.size, solution.size)
        target = librosa.util.fix_length(target, size=length)
        solution = librosa.util.fix_length(solution, size=length)

        frame_length = 4000  # 4000 samples ~ 90 ms
        target_frames = librosa.util.frame(target,
                                           frame_length=frame_length,
                                           hop_length=frame_length,
                                           axis=0)
        solution_frames = librosa.util.frame(solution,
                                             frame_length=frame_length,
                                             hop_length=frame_length, axis=0)

        distances = []
        target_energy = []
        assert target_frames.size == solution_frames.size
        for i in range(target_frames.shape[0]):
            distance = custom_distance_metric(target_frames[i], solution_frames[i])
            distances.append(distance)
            energy = np.sum(np.square(target_frames[i]))
            target_energy.append(energy)
        distances = np.array(distances)
        energy = np.array(target_energy)
        weighted_sum = np.sum(distances * energy) / np.sum(energy)
        return weighted_sum.item()
    return custom_distance

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
