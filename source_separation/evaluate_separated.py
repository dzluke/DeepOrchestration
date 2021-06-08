import os
import argparse

import numpy as np
import json
import librosa
import museval

from pipeline import separation_models, SAMPLING_RATE


def main(targets_path, subtargets_path, separated_path, output_path):
    # Get metadata
    metadata_path = os.path.join(targets_path, 'metadata.json')
    with open(metadata_path, 'r') as metadata_file:
        metadata = json.load(metadata_file)

    # Initialize results dictionary.
    results = {}
    for model in separation_models:
        results[model] = []

    # Evaluate each target
    for target in metadata:
        print('Evaluating results for {}'.format(target))
        # Get reference sources (= padded subtargets)
        target_metadata = metadata[target]
        ref_sources = []
        for data in target_metadata:
            audio, _ = librosa.load(os.path.join(subtargets_path,
                                                 data['name'] + '.wav'),
                                    sr=SAMPLING_RATE)
            audio = np.pad(audio, data['padding'])
            ref_sources.append(audio)

        # WORKAROUND (remove eventually)
        target = target.replace('*', '_')

        # Evaluate for each separation model
        for model in separation_models:
            # Get estimated sources(/subtargets)
            est_path = os.path.join(separated_path, model, target)
            est_sources_files = librosa.util.find_files(est_path)
            if len(est_sources_files) != len(target_metadata):
                raise ValueError('The number of estimated source should match'
                                 'the number of references sources.')
            est_sources = []
            for file in est_sources_files:
                audio, _ = librosa.load(file, sr=SAMPLING_RATE)
                est_sources.append(audio)

            (sdr, isr, sir, sar, perm) = museval.metrics.bss_eval(
                ref_sources, est_sources, compute_permutation=True)
            print('\t{}\tMean sdr: {}'.format(model, sdr.mean()))
            results[model].append([target, sdr, sir, sar, perm])

    if output_path:
        # Save results (as pandas DataFrame I think)
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('targets_path', type=str)
    parser.add_argument('subtargets_path', type=str)
    parser.add_argument('separated_path', type=str)
    parser.add_argument('-o', '--out_path', type=str, default=None,
                        dest='out_path')
    args, _ = parser.parse_known_args()
    main(args.targets_path, args.subtargets_path, args.separated_path,
         args.out_path)
