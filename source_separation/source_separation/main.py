# This is the main script to run experiments with one command!
import json
import os
from pathlib import Path
from argparse import ArgumentParser
from configparser import ConfigParser
from shutil import copyfile

import librosa

from orchestration import (
    clean_orch_config,
    set_config_parameter,
    orchestrate_with_threshold,
    assign_orchestras,
    save_best_orchestration
)
from utils import frame_distance, spectral_distance, create_combinations

# Get configuration
config = ConfigParser(inline_comment_prefixes="#")
config.read("config.ini")


def new_main_because_i_cant_even(ds_path, n_sources):

    pass

def main(ds_path, n_sources):
    distance_metric = frame_distance(
        spectral_distance
    )

    num_completed = 0
    # full_target_distances is a nested list of distances of orchestrating full targets without separation
    # full_target_distances[i][k] is the distance between target i and the orchestration of target i with threshold k
    full_target_distances = []
    # distances for separated targets
    # separated_target_distances['model'][i] is a list of length 81 of the distances for all possible combinations (3^4)
    # of target i after being separated by 'model', orchestrated, then combined
    separated_target_distances = {}
    for model in config["separation"]["methods"].split(", "):
        separated_target_distances[model] = []
    # ground_truth_distances[i] is a list of length 81, it is the distances for all possible combinations of
    # orchestrations of the ground truth source subtargets
    ground_truth_distances = []

    # a dictionary that represents the current state of the pipeline, including how many targets have
    # been completed, and the distances.
    # this is stored/loaded as a json so the process can be stopped and restarted

    results = {
        "num_completed": num_completed,
        "full_target_distances": full_target_distances,
        "separated_target_distances": separated_target_distances,
        "ground_truth_distances": ground_truth_distances,
    }

    # load saved results from json, or create json if none exists
    results_path = config["paths"]["results_path"]
    try:
        results_file = open(results_path, "r")
        results = json.load(results_file)
        if len(results) != 0:  # if it's not an empty json
            num_completed = results["num_completed"]
            full_target_distances = results["full_target_distances"]
            separated_target_distances = results["separated_target_distances"]
            ground_truth_distances = results["ground_truth_distances"]
            print("Backing up results.json to results_backup.json")
            copyfile(results_path, "results_backup.json")
    except FileNotFoundError:
        results_file = open(results_path, "w")
    results_file.close()

    targets = librosa.util.find_files(
        os.path.join(ds_path, "targets", f"{n_sources}sources")
    )
    print("Database contains {} targets".format(len(targets)))

    if config["orchestration"].getint(
        "num_orchestrations_to_save"
    ) > 0 and not os.path.exists(config["paths"]["saved_orchestrations_path"]):
        os.mkdir(config["paths"]["saved_orchestrations_path"])

    sample_rate = config["audio"].getint("sample_rate")
    while num_completed < len(targets):
        target_path = targets[num_completed]
        target, _ = librosa.load(target_path, sr=sample_rate)
        target_name = os.path.basename(target_path)
        target_name = os.path.splitext(target_name)[0]
        save_path = Path(config["paths"]["all_orchestrations_path"]) / target_name

        print("Target:", target_name)
        print("num completed:", num_completed)

        clean_orch_config(
            config["paths"]["config_path"],
            config["paths"]["sol_path"],
            config["paths"]["analysis_db_path"],
        )

        # orchestrate full (non-separated) target with Orchidea
        print("Orchestrating full target")
        set_config_parameter(
            config["paths"]["config_path"],
            "orchestra",
            config["orchestration"]["full_orchestra"].split(", "),
        )
        full_target_orchestrations = []
        distances = []
        save_path = save_path / "full_orchestration"

        orchestrations = orchestrate_with_threshold(target, save_path)
        full_target_orchestrations = orchestrations
        for orch in orchestrations:
            distance = distance_metric(target, orch)
            distances.append(distance)
        full_target_distances.append(distances)
        save_path = save_path.parent

        # separate target into subtargets using different separator functions
        print("Separating target into subtargets")
        # all_subtargets maps a model name to a list of subtargets
        all_subtargets = {}
        for model, separator in separation_functions.items():
            subtargets, _ = separator(target_path, n_sources)
            all_subtargets[model] = subtargets

        # orchestrate subtargets with different segmentation thresholds
        print("Orchestrating subtargets with different thresholds")
        orchestras = assign_orchestras(
            n_sources, config["orchestration"]["full_orchestra"].split(", ")
        )
        # orchestrated_subtargets[model][j][k] is
        # the jth subtarget, separated via 'model', orchestrated with threshold k
        orchestrated_subtargets = {}
        for model in separation_functions.keys():
            orchestrated_subtargets[model] = [[] for _ in range(n_sources)]

        for (
            model,
            subtargets,
        ) in all_subtargets.items():  # for each separation algorithm
            save_path = save_path / model
            for j in range(len(subtargets)):  # for each subtarget
                subtarget = subtargets[j]
                orchestra = orchestras[j]
                set_config_parameter(
                    config["paths"]["config_path"], "orchestra", orchestra
                )
                save_name = "subtarget" + str(j)
                save_path = save_path / save_name
                orchestrations = orchestrate_with_threshold(subtarget, save_path)
                orchestrated_subtargets[model][j] = orchestrations
                save_path = save_path.parent
            save_path = save_path.parent

        # create all possible combinations of orchestrated subtargets and calculate distance
        # print("Combining subtargets and calculating distance")
        combinations = {}
        for model, subtargets in orchestrated_subtargets.items():
            # for each separation algorithm
            combinations[model] = create_combinations(subtargets)
            distances = []
            for orch in combinations[model]:
                distance = distance_metric(target, orch)
                distances.append(distance)
            separated_target_distances[model].append(distances)

        # calculate ground truth
        print("Orchestrating ground truth")
        # read metadata
        with open(TARGET_METADATA_PATH, "r") as metadata_file:
            metadata = json.load(metadata_file)
        metadata = metadata[target_name]
        sources = []
        for data in metadata:
            audio, _ = librosa.load(data["path"], sr=sample_rate)
            audio = np.pad(audio, data["padding"])
            sources.append(audio)
        if len(sources) != n_sources:
            if len(sources) == 2:
                # this biases the results
                orchestras[0] += orchestras[2]
                orchestras[1] += orchestras[3]
                orchestras = orchestras[:2]
            else:
                raise Exception(
                    "Code is not ready to handle {} sources".format(len(sources))
                )
        ground_truth_solutions = [[] for _ in range(len(sources))]
        save_path = save_path / "ground_truth"
        for i in range(len(sources)):
            source = sources[i]
            orchestra = orchestras[i]
            set_config_parameter(config["paths"]["config_path"], "orchestra", orchestra)
            save_name = "source" + str(i)
            save_path = save_path / save_name

            orchestrations = orchestrate_with_threshold(source, save_path)
            ground_truth_solutions[i] = orchestrations
            save_path = save_path.parent
        save_path = save_path.parent

        ground_truth_combinations = create_combinations(ground_truth_solutions)
        distances = []
        for orch in ground_truth_combinations:
            distance = distance_metric(target, orch)
            distances.append(distance)
        ground_truth_distances.append(distances)

        if num_orchestrations_to_save > 0:
            orch_folder_path = os.path.join(
                config["paths"]["saved_orchestrations_path"], target_name
            )
            if not os.path.exists(orch_folder_path):
                os.mkdir(orch_folder_path)
            # save target
            copyfile(target_path, os.path.join(orch_folder_path, target_name + ".wav"))
            # save best full target orchestration
            distances = full_target_distances[-1]
            name = os.path.join(orch_folder_path, "full_orchestration.wav")
            save_best_orchestration(full_target_orchestrations, distances, name)
            # save best separated orchestration
            for model, combos in combinations.items():
                distances = separated_target_distances[model][-1]
                name = os.path.join(orch_folder_path, model + "_orchestration.wav")
                save_best_orchestration(combos, distances, name)
            # save best ground truth orchestration
            distances = ground_truth_distances[-1]
            name = os.path.join(orch_folder_path, "ground_truth_orchestration.wav")
            save_best_orchestration(ground_truth_combinations, distances, name)
            # save distances in .txt file
            with open(os.path.join(orch_folder_path, "distances.txt"), "w") as f:
                f.write("full target distance: {}\n".format(full_target_distances[-1]))
                f.write(
                    "ground truth distance: {}\n".format(ground_truth_distances[-1])
                )
                for model, combos in combinations.items():
                    f.write(
                        "{} distance: {}\n".format(
                            model, separated_target_distances[model][-1]
                        )
                    )
            # move source separation results
            separations_folder = os.path.join(orch_folder_path, "separations")
            if not os.path.exists(separations_folder):
                os.mkdir(separations_folder)
            for model_name in config["separation"]["methods"].split(", "):
                # WORKAROUND (remove eventually)
                target_name = target_name.replace("*", "_")
                folder = os.path.join(SEPARATIONS_PATH, model_name, target_name)
                if not os.path.exists(os.path.join(separations_folder, model_name)):
                    os.mkdir(os.path.join(separations_folder, model_name))
                for source in os.listdir(folder):
                    source_path = os.path.join(folder, source)
                    copyfile(
                        source_path,
                        os.path.join(separations_folder, model_name, source),
                    )
            num_orchestrations_to_save -= 1

        num_completed += 1
        # save results to json
        results["num_completed"] = num_completed
        results["full_target_distances"] = full_target_distances
        results["separated_target_distances"] = separated_target_distances
        results["ground_truth_distances"] = ground_truth_distances
        with open(config["paths"]["results_path"], "w") as f:
            json.dump(results, f, indent="\t")

        # remove temp files created during separation
        # for model_name in separation_models:
        #     path = os.path.join(SEPARATIONS_PATH, model_name, target_name)
        #     if os.path.exists(path):
        #         remove_directory(path)

    # remove files created during pipeline
    for file in ["target.wav", "segments.txt"]:
        if os.path.exists(file):
            os.remove(file)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("ds_path", type=str)
    parser.add_argument("--n-sources", type=int, default=4)
    args, _ = parser.parse_known_args()
    main(args.ds_path, args.n_sources)
