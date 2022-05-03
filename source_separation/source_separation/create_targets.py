# This script is used to combine samples into targets.
import argparse
import json
import os
from configparser import ConfigParser
from pathlib import Path
from random import randint, sample

import librosa
import soundfile as sf
from scipy.special import comb
from tqdm import tqdm

from utils import clear_directory, combine_with_offset

# Get configuration
config = ConfigParser(inline_comment_prefixes="#")
config.read("config.ini")


def create_targets(combinations, outdir, padding=False):
    """
    All samples in paths[i] are combined into a single target and written as
    .wav's to TARGETS_PATH
    :param paths: A nested list of paths to samples, ex:
    [['Beethoven_chord1.wav', 'drops.wav], ['gong.wav', 'bell.wav']]
    :return:
    """
    metadata = {}

    for sample_paths in tqdm(combinations):
        samples = []
        longest_sample = None
        longest_length = 0
        for path in sample_paths:
            sample, _ = librosa.load(
                path,
                sr=config["audio"].getint("sample_rate"),
                duration=config["targets"].getint("max_duration"),
            )
            base = os.path.basename(path)
            name = os.path.splitext(base)[0]
            sample_metadata = {"sample name": name, "audio": sample}
            samples.append(sample_metadata)
            if sample.size > longest_length:
                longest_sample = sample_metadata
                longest_length = sample.size
        # calculate padding needed
        for sample_metadata in samples:
            if sample_metadata == longest_sample or not padding:
                sample_metadata["padding"] = [0]
            else:
                sample_metadata["padding"] = [randint(1, longest_length // 2)]
        # recalculate longest length to include offset
        for sample_metadata in samples:
            longest_length = max(
                longest_length,
                sample_metadata["audio"].size + sample_metadata["padding"][0],
            )
        # calculate end padding
        for sample_metadata in samples:
            pad = longest_length - (
                sample_metadata["padding"][0] + sample_metadata["audio"].size
            )
            sample_metadata["padding"].append(pad)

        # Combine samples into a single audio
        target = combine_with_offset(samples)
        name = [sample_metadata["sample name"] for sample_metadata in samples]
        name = "&".join(sorted(name))

        # Add informations to metadata file
        for sample in samples:
            sample.pop("audio")  # we don't need to store the audio anymore
        metadata[name] = samples

        # write to folder
        target_fname = os.path.join(outdir, name + ".wav")
        sf.write(target_fname, target, samplerate=config["audio"].getint("sample_rate"))

    # write metadata
    metadata_file = os.path.join(outdir, "targets_metadata.json")
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)


def main(ds_path, n_sources, num_targets):
    targets_path = os.path.join(ds_path, "targets", f"{n_sources}sources")
    Path(targets_path).mkdir(parents=True, exist_ok=True)
    clear_directory(targets_path)
    samples = librosa.util.find_files(os.path.join(ds_path, "samples"))

    sample_paths = set()
    if num_targets > comb(len(samples), n_sources):
        raise ValueError("There are not enough samples to make that many combinations!")

    while len(sample_paths) < num_targets:
        paths = sample(samples, n_sources)
        paths = sorted(paths)
        paths = tuple(paths)  # you can't add lists to a set
        sample_paths.add(paths)

    print("Creating {} targets".format(len(sample_paths)))
    create_targets(sample_paths, targets_path)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ds_path", type=str)
    parser.add_argument(
        "--n-sources",
        type=int,
        help="Number of samples to combine to get a target.",
        default=config["separation"].getint("n_sources"),
    )
    parser.add_argument(
        "--num-targets",
        type=int,
        help="Number of targets to create.",
        default=config["targets"].getint("num_targets"),
    )
    args, _ = parser.parse_known_args()

    main(args.ds_path, args.n_sources, args.num_targets)
