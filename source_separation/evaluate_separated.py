import argparse
import csv
import json
import os

import librosa
import mir_eval
import museval
import numpy as np

from pipeline import SAMPLING_RATE, separation_models


def main(metadata_path, subtargets_path, separated_path, outdir=None):

    # Get metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    # Initialize results dictionary.
    results = {}
    for model in separation_models:
        results[model] = []

    # Evaluate each target
    for idx, target in enumerate(metadata):
        print('{}/{} Evaluating results for {}'.format(idx, len(metadata),
                                                       target))
        # Get reference sources (= padded subtargets)
        target_metadata = metadata[target]
        ref_sources = []
        for data in target_metadata:
            audio, _ = librosa.load(os.path.join(subtargets_path,
                                                 data['name'] + '.wav'),
                                    sr=SAMPLING_RATE)
            audio = librosa.util.normalize(audio)
            audio = np.pad(audio, data['padding'])
            ref_sources.append(audio)
        ref_sources = np.array(ref_sources)
        # ref_sources = ref_sources[..., np.newaxis]  #probably unnecessary

        # WORKAROUND (remove eventually)
        target = target.replace('*', '_')

        # Evaluate for each separation model
        for model in separation_models:
            # Get estimated sources(/subtargets)
            est_path = os.path.join(separated_path, model, target)
            est_sources_files = librosa.util.find_files(est_path)
            assert len(est_sources_files) == len(target_metadata)

            est_sources = []
            for file in est_sources_files:
                audio, _ = librosa.load(file, sr=SAMPLING_RATE)
                audio = librosa.util.normalize(audio)
                est_sources.append(audio)
            est_sources = np.array(est_sources)
            # est_sources = est_sources[..., np.newaxis]  #probably unnecessary

            (sdr, isr, sir, sar, perm) = museval.metrics.bss_eval(
                ref_sources, est_sources, compute_permutation=True,
                window=ref_sources.shape[1])
            print('\t{}\t Mean sdr: {}'.format(model, np.nanmean(sdr)))
            results[model].append([target, sdr, sir, sar, perm])

    if outdir is not None:
        with open(os.path.join(outdir, "results.csv"), 'w', newline='\n') as f:
            wr = csv.writer(f, quoting=csv.QUOTE_ALL)
            wr.writerows(results)
    return 0



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('metadata_file', type=str)
    parser.add_argument('subtargets_path', type=str)
    parser.add_argument('separated_path', type=str)
    parser.add_argument('-o', '--out_path', type=str, default=None,
                        dest='out_path')
    args, _ = parser.parse_known_args()
    main(args.metadata_file, args.subtargets_path, args.separated_path)
