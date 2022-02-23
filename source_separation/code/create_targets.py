# This script is used to combine subtargets into targets
import argparse
import json
import os
import random
from configparser import ConfigParser

import librosa
import soundfile as sf

from .utils import clear_directory, combine_with_offset

# Get configuration
config = ConfigParser()
config.read("config.ini")


def create_targets(paths):
    """
    All samples in paths[i] are combined into a single target and written as
    .wav's to TARGETS_PATH
    :param paths: A nested list of paths to samples, ex:
    [['Beethoven_chord1.wav', 'drops.wav], ['gong.wav', 'bell.wav']]
    :return:
    """
    targets = []
    names = []
    metadata = {}

    for sample_paths in paths:
        samples = []
        longest_sample = None
        longest_length = 0
        for path in sample_paths:
            sample, _ = librosa.load(path,
                                     sr=config['audio'].getint('sample_rate'))
            base = os.path.basename(path)
            name = os.path.splitext(base)[0]
            sample_metadata = {'name': name, 'path': path, 'audio': sample}
            samples.append(sample_metadata)
            if sample.size > longest_length:
                longest_sample = sample_metadata
                longest_length = sample.size
        # calculate padding needed
        for sample_metadata in samples:
            if sample_metadata == longest_sample:
                sample_metadata['padding'] = [0]
            else:
                sample_metadata['padding'] = [random.randint(1, longest_length // 2)]
        # recalculate longest length to include offset
        for sample_metadata in samples:
            longest_length = max(longest_length, sample_metadata['audio'].size
                                 + sample_metadata['padding'][0])
        # calculate end padding
        for sample_metadata in samples:
            pad = longest_length - (sample_metadata['padding'][0]
                                    + sample_metadata['audio'].size)
            sample_metadata['padding'].append(pad)
        combination = combine_with_offset(samples)
        targets.append(combination)
        name = [sample_metadata['name'] for sample_metadata in samples]
        name = '+'.join(sorted(name))
        for sample in samples:
            sample.pop('audio')  # we don't need to store the audio anymore
        metadata[name] = samples
        names.append(name)
    # write to folder
    for i in range(len(targets)):
        target = targets[i]
        name = names[i]
        path = os.path.join(config['paths']['targets_path'], name) + '.wav'
        if not os.path.exists(path):
            sf.write(path, target,
                     samplerate=config['audio'].getint('sample_rate'))
    # write metadata
    with open(config['paths']['target_metadata_path'], 'w') as metadata_file:
        json.dump(metadata, metadata_file, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("num-subtargets",
                        type=int,
                        help="Number of subtargets (samples) to combine to get \
                            a target.",
                        default=4)
    parser.add_argument("--num-targets",
                        type=int,
                        help="Number of targets to create.",
                        default=config["misc"]["n_targets"])
    args, _ = parser.parse_known_args()

    clear_directory(config['paths']['targets_path'])
    samples = librosa.util.find_files(config['paths']['sample_database'])
    sample_paths = set()
    while len(sample_paths) < args.num_targets:
        paths = random.sample(samples, args.num_subtargets)
        paths = sorted(paths)
        paths = tuple(paths)  # you can't add lists to a set
        sample_paths.add(paths)
    print("Creating {} targets".format(len(sample_paths)))
    create_targets(sample_paths)
    print("Done.")
