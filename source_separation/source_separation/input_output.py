# This contains various I/O function
import os

import librosa
import numpy as np

def load_audio_files(paths, sample_rate, normalize=True, paddings=None):
    """ Load a list of paths to audio files and return an array.

    Args:
        paths (list): List of paths to the files.
        normalize (bool, optional): If True, normalize the audio.
    """
    waveforms = []
    for i, file in enumerate(paths):
        audio, _ = librosa.load(file, sr=sample_rate)
        if normalize:
            audio = librosa.util.normalize(audio)
        if paddings:
            audio = np.pad(audio, paddings[i])
        waveforms.append(audio)
    return waveforms


class Target():
    """Inspired by MSAF
    Holds every paths stuff for a given target.
    """

    def __init__(self, ds_path, target_name, n_sources):
        self.ds_path = ds_path
        self.name = target_name
        self.n_sources = n_sources
        self.target_metadata = 0 # target_metadata  #TODO: load metadata in json file

    def get_samples_paths(self):
        files = []
        for data in self.target_metadata:
            f = os.path.join(self.ds_path, 'samples', data['name'] + '.wav')
            files.append(f)
        return files

    def get_padding(self):
        paddings = []
        for data in self.target_metadata:
            paddings.append(data['padding'])
        return paddings

    def get_separated_paths(self, model_name):
        sep_folder = os.path.join(self.ds_path, 'separated',
                                  f'{self.n_sources}sources',
                                  model_name,
                                  self.target_name)
        files = librosa.util.find_files(sep_folder)
        return files

    def path(self):
        """Mixture? Target? Which one should I use?
        """
        file = os.path.join(self.ds_path, 'targets',
                            f'{self.n_sources}sources',
                            self.target_name + '.wav')
        return file
