# This script is used to evaluate the quality of the source separation stage.
import json
import os
from argparse import ArgumentParser
from configparser import ConfigParser
from pathlib import Path

import mir_eval
import numpy as np
from tqdm import tqdm

from input_output import TargetFileStruct, load_audio_files, rename_separated

# Get configuration
config = ConfigParser(inline_comment_prefixes="#")
config.read("config.ini")
RESULTS_FNAME = "sep_results.json"


def eval_target(target_file_struct, separation_models):
    sample_rate = config["audio"].getint("sample_rate")
    source_waveforms = load_audio_files(
        target_file_struct.get_samples_paths(),
        sample_rate,
        normalize=True,
        paddings=target_file_struct.get_padding(),
    )
    source_waveforms = np.stack(source_waveforms)

    # Evaluate for each separation model
    results = {}
    for model in separation_models:
        separated_waveforms = load_audio_files(
            target_file_struct.get_separated_paths(model), sample_rate, normalize=True
        )
        separated_waveforms = np.stack(separated_waveforms)

        sdr, sir, sar, perm = mir_eval.separation.bss_eval_sources(
            source_waveforms, separated_waveforms
        )
        # NB: estimated source number perm[j] corresponds to true source number j
        print("\t{}\t Average SDR: {}".format(model, np.mean(sdr)))
        results[model] = {
            "sdr": str(np.mean(sdr)),
            "sir": str(np.mean(sir)),
            "sar": str(np.mean(sar)),
            "permutation": perm.tolist(),
        }
    return results


def evaluate_separation(metadata_path, ds_path, results_file, rename_files=True):
    separation_methods = config["separation"]["methods"].split(", ")

    # Load targets metadata file
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    # Initialize results dictionary.
    results = init_results_dict(results_file)

    # Evaluate each target
    for idx, target_name in enumerate(tqdm(metadata)):
        if target_name in results:
            continue  # skip if this target has already been evaluated

        print("{}/{} Evaluating results for {}".format(idx, len(metadata), target_name))
        target_file_struct = TargetFileStruct(
            ds_path, target_name, target_metadata=metadata[target_name]
        )

        results[target_name] = eval_target(target_file_struct, separation_methods)
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        if rename_files:
            for method in separation_methods:
                rename_separated(
                    target_file_struct,
                    permutation=results[target_name][method]["permutation"],
                    separation_method=method,
                )

    prints_results(separation_methods, results)


def init_results_dict(results_file):
    if os.path.isfile(results_file):
        with open(results_file, "r") as f:
            results = json.load(f)
    else:
        results = {}
    return results


def prints_results(separation_models, results):
    average_results = {model: [] for model in separation_models}
    for target in results:
        for model in separation_models:
            average_results[model].append(results[target][model]["sdr"])
    print("Results over all targets:")
    for model in separation_models:
        print(model, np.mean([float(i) for i in average_results[model]]))
        print(
            f"\tStandard deviation", np.std([float(i) for i in average_results[model]])
        )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("ds_path", type=str)
    parser.add_argument(
        "--n-sources", type=int, default=config["separation"].getint("n_sources"),
    )
    args, _ = parser.parse_known_args()

    metadata_file = os.path.join(
        args.ds_path, "targets", f"{args.n_sources}sources", "targets_metadata.json"
    )
    outdir = os.path.join(args.ds_path, "separated", f"{args.n_sources}sources")
    Path(outdir).mkdir(parents=True, exist_ok=True)
    results_file = os.path.join(outdir, RESULTS_FNAME)

    evaluate_separation(metadata_file, args.ds_path, results_file)
