# This contains various I/O function
import os
from configparser import ConfigParser

import librosa
import numpy as np

from utils import remove_extension

# Get configuration
config = ConfigParser(inline_comment_prefixes="#")
config.read("config.ini")


def load_audio_files(paths, sample_rate, normalize=True, paddings=None):
    """ Load a list of paths to audio files and return an array.

    Args:
        paths (list): List of paths to the files.
        normalize (bool, optional): If True, normalize the audio.
    """
    waveforms = []
    for i, file in enumerate(paths):
        audio, _ = librosa.load(
            file, sr=sample_rate, duration=config["targets"].getint("max_duration")
        )
        if normalize:
            audio = librosa.util.normalize(audio)
        if paddings:
            audio = np.pad(audio, paddings[i])
        waveforms.append(audio)
    return waveforms


def rename_separated(target_file_struct, permutation, separation_method):
    samples_paths = target_file_struct.get_samples_paths()
    separated_paths = target_file_struct.get_separated_paths(separation_method)
    separated_folder = os.path.dirname(separated_paths[0])

    for i, sample in enumerate(samples_paths):
        sample_name = remove_extension(os.path.basename(sample))
        renamed_sep = os.path.join(separated_folder, sample_name + "_sep.wav")
        os.rename(separated_paths[permutation[i]], renamed_sep)


class TargetFileStruct:
    """ Holds every paths for a given target.

    Inspired by MSAF
    """

    def __init__(self, ds_path, target_name, target_metadata=None, n_sources=None):
        self.ds_path = ds_path
        self.target_name = target_name

        assert target_metadata or n_sources
        if target_metadata:
            self.target_metadata = target_metadata
            self.n_sources = len(target_metadata)
        else:
            self.n_sources = n_sources

    def get_path(self):
        """Get path to the target file."""
        file = os.path.join(
            self.ds_path,
            "targets",
            f"{self.n_sources}sources",
            self.target_name + ".wav",
        )
        return file

    def get_full_orch_path(self):
        """Get path to the full orchestration of the target (no separation)."""
        path = os.path.join(
            self.ds_path,
            "orchestrated",
            f"{self.n_sources}sources",
            self.target_name,
            "full_target_orch.wav",
        )
        return path

    def get_samples_paths(self):
        """Get paths to all samples used to create the target."""
        files = []
        for data in self.target_metadata:
            f = os.path.join(self.ds_path, "samples", data["name"] + ".wav")
            files.append(f)
        return sorted(files)

    def get_padding(self):
        """Get the samples paddings used to create the target."""
        paddings = [[] for i in range(self.n_sources)]
        # We have a problem, we need to make sure that the paddings are aligned with the correct
        # sample. (this is a problem only with previous versions of the dataset)
        args_sorted_samples = np.argsort([s['name'] for s in self.target_metadata])
        for i, data in enumerate(self.target_metadata):
            # paddings.append(data["padding"])
            paddings[args_sorted_samples[i]] = data['padding']
        return paddings

    def get_separated_paths(self, separation_method):
        """Get paths of the separated waveforms outputted by separation_method."""
        sep_folder = os.path.join(
            self.ds_path,
            "separated",
            f"{self.n_sources}sources",
            separation_method,
            self.target_name,
        ).replace('*', '_')
        files = librosa.util.find_files(sep_folder)
        assert len(files) == self.n_sources
        return sorted(files)

    def get_orch_folder(self):
        orch_folder = os.path.join(
            self.ds_path, "orchestrated", f"{self.n_sources}sources", self.target_name
        )
        return orch_folder
