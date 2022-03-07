# This script is used to combine subtargets into targets
import argparse
import json
import os
from configparser import ConfigParser
from random import randint, sample

import librosa
import soundfile as sf
from tqdm import tqdm

from utils import clear_directory, combine_with_offset

# Get configuration
config = ConfigParser(inline_comment_prefixes="#")
config.read("config.ini")


def create_targets(paths):
    """
    All samples in paths[i] are combined into a single target and written as
    .wav's to TARGETS_PATH
    :param paths: A nested list of paths to samples, ex:
    [['Beethoven_chord1.wav', 'drops.wav], ['gong.wav', 'bell.wav']]
    :return:
    """
    metadata = {}

    for sample_paths in tqdm(paths):
        samples = []
        longest_sample = None
        longest_length = 0
        for path in sample_paths:
            sample, _ = librosa.load(path,
                                     sr=config['audio'].getint('sample_rate'))
            base = os.path.basename(path)
            name = os.path.splitext(base)[0]
            sample_metadata = {'name': name,
                               'path': path,
                               'audio': sample}
            samples.append(sample_metadata)
            if sample.size > longest_length:
                longest_sample = sample_metadata
                longest_length = sample.size
        # calculate padding needed
        for sample_metadata in samples:
            if sample_metadata == longest_sample:
                sample_metadata['padding'] = [0]
            else:
                sample_metadata['padding'] = [randint(1, longest_length // 2)]
        # recalculate longest length to include offset
        for sample_metadata in samples:
            longest_length = max(longest_length, sample_metadata['audio'].size
                                 + sample_metadata['padding'][0])
        # calculate end padding
        for sample_metadata in samples:
            pad = longest_length - (sample_metadata['padding'][0]
                                    + sample_metadata['audio'].size)
            sample_metadata['padding'].append(pad)

        # Combine samples into a single audio
        target = combine_with_offset(samples)
        name = [sample_metadata['name'] for sample_metadata in samples]
        name = '+'.join(sorted(name))

        # Add informations to metadata file
        for sample in samples:
            sample.pop('audio')  # we don't need to store the audio anymore
        metadata[name] = samples

        # write to folder
        target_fname = os.path.join(config['paths']['targets'],
                                    f"{len(samples)}sources",
                                    name + '.wav')
        sf.write(target_fname, target,
                 samplerate=config['audio'].getint('sample_rate'))

    # write metadata
    metadata_file = os.path.join(config['paths']['targets'],
                                 f"{len(samples)}sources",
                                 "metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples",
                        type=int,
                        help="Number of samples to combine to get a target.",
                        default=config['separation'].getint('num_samples'))
    parser.add_argument("--num-targets",
                        type=int,
                        help="Number of targets to create.",
                        default=config["misc"].getint("num_targets"))
    args, _ = parser.parse_known_args()


    targets_path = os.path.join(config['paths']['targets'],
                                f"{args.num_samples}sources")
    clear_directory(targets_path)
    samples = librosa.util.find_files(config['paths']['samples'])
    sample_paths = set()
    while len(sample_paths) < args.num_targets:
        paths = sample(samples, args.num_samples)
        paths = sorted(paths)
        paths = tuple(paths)  # you can't add lists to a set
        sample_paths.add(paths)
    print("Creating {} targets".format(len(sample_paths)))
    create_targets(sample_paths)
    print("Done.")
