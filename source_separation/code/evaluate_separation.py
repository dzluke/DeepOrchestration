import csv
import json
import os
from multiprocessing import Process, Manager
from argparse import ArgumentParser
from configparser import ConfigParser

import librosa
import mir_eval
import museval
import numpy as np

# Get configuration
config = ConfigParser(inline_comment_prefixes="#")
config.read("config.ini")


def eval_one_target(subtargets_path, separated_path, separation_models,
                    sample_rate, metadata, results, target):
    target_metadata = metadata[target]
    ref_sources = []
    for data in target_metadata:
        audio, _ = librosa.load(os.path.join(subtargets_path,
                                             data['name'] + '.wav'),
                                sr=sample_rate)
        audio = librosa.util.normalize(audio)
        audio = np.pad(audio, data['padding'])
        ref_sources.append(audio)
    ref_sources = np.array(ref_sources)
    # ref_sources = ref_sources[..., np.newaxis]  #probably unnecessary

    # WORKAROUND (remove eventually)
    # this is a workaround because when the separations were performed, the target names were incorrectly concatenated
    # instead of being joined with a "*", they were joined with a "_"
    target = target.replace('*', '_')

    # Evaluate for each separation model
    for model in separation_models:
        # Get estimated sources(/subtargets)
        est_path = os.path.join(separated_path, model, target)
        est_sources_files = librosa.util.find_files(est_path)
        assert len(est_sources_files) == len(target_metadata)

        est_sources = []
        for file in est_sources_files:
            audio, _ = librosa.load(file, sr=sample_rate)
            audio = librosa.util.normalize(audio)
            est_sources.append(audio)
        est_sources = np.array(est_sources)
        # est_sources = est_sources[..., np.newaxis]  #probably unnecessary

        (sdr, isr, sir, sar, perm) = museval.metrics.bss_eval(
                ref_sources, est_sources, compute_permutation=True,
                window=ref_sources.shape[1])
        print('\t{}\t Mean sdr: {}'.format(model, np.nanmean(sdr)))
        results.append([target, model, sdr, sir, sar, perm])


def main(metadata_path, subtargets_path, separated_path, outdir=None,
         jobs=None):
    separation_models = config['separation']['methods'].split(", ")
    sample_rate = config['audio'].getint('sample_rate')

    # Get metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    # Initialize results dictionary.
    if jobs:
        manager = Manager()
        results = manager.list()
        processes = []
        for idx, target in enumerate(metadata):
            p = Process(target=eval_one_target,
                        args=(subtargets_path, separated_path,
                              separation_models, sample_rate, metadata,
                              results, target)
                        )
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
    else:
        results = []
        # Evaluate each target
        for idx, target in enumerate(metadata):
            print('{}/{} Evaluating results for {}'.format(idx, len(metadata),
                                                        target))
            # Get reference sources (= padded subtargets)
            eval_one_target(subtargets_path, separated_path, separation_models,
                            sample_rate, metadata, results, target)

    if outdir is not None:
        with open(os.path.join(outdir, "eval_results.csv"), 'w', newline='\n') as f:
            wr = csv.writer(f, quoting=csv.QUOTE_ALL)
            wr.writerows(results)
    return 0

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('metadata_file', type=str)
    parser.add_argument('subtargets_path', type=str)
    parser.add_argument('separated_path', type=str)
    parser.add_argument('-o', '--outdir', type=str, default=None)
    parser.add_argument("-j", "--jobs", type=int, default=None,
                        help="Number of jobs for multiprocessing.")
    args, _ = parser.parse_known_args()
    main(args.metadata_file, args.subtargets_path, args.separated_path,
         args.outdir, args.jobs)
