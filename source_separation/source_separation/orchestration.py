# This contains the necessary to run target-based assisted orchestation with
# the Orchidea framework, available at:
#   https://www.orch-idea.org
#
import os
import random
import subprocess
from argparse import ArgumentParser
from configparser import ConfigParser
from shutil import copyfile

import librosa
import soundfile as sf
from tqdm import tqdm
from input_output import TargetFileStruct

from utils import cosine_distance, frame_distance, set_config_parameter, remove_extension

# Get configuration
config = ConfigParser(inline_comment_prefixes="#")
config.read("config.ini")


def orchestrate(target, sample_rate, config_path):
    """
    Orchestrate 'target' using Orchidea
    :param target: sound file as numpy array, shape (len,)
    :return: orchestrated solution as numpy array, shape (len,)
    """
    sf.write("target.wav", target, sample_rate)
    if config["orchestration"].getboolean("darling"):
        cmd = ["darling", "shell", "./orchestrate", "target.wav", config_path]
    else:
        cmd = ["./orchestrate", "target.wav", config_path]
    subprocess.run(cmd, stdout=subprocess.DEVNULL)  # this suppresses output
    solution, _ = librosa.load("connection.wav", sr=sample_rate)
    return solution


def assign_orchestras(num_orchestras, full_orchestra):
    """
    Randomly divide the full orchestra into 'num_orchestras' sub-orchestras
    :param num_orchestras: the number of orchestras you want the full orchestra divided into
    :return orchestras: a nested list of orchestras where len(orchestras) == num_orchestras
    """
    orchestras = (
        []
    )  # nested list of orchestras, orchestras[i] is the orchestra for segment i
    available_instruments = full_orchestra.copy()
    orchestra_size = len(available_instruments) // num_orchestras
    extra_instruments = len(available_instruments) % num_orchestras

    random.shuffle(available_instruments)

    for i in range(num_orchestras):
        orchestra = available_instruments[:orchestra_size]
        for instrument in orchestra:
            available_instruments.remove(instrument)
        orchestras.append(orchestra)

    for i in random.sample(range(num_orchestras), extra_instruments):
        instrument = available_instruments.pop()
        orchestras[i].append(instrument)

    return orchestras


def orchestrate_with_threshold(
    waveform, save_path, thresholds, config_path, sample_rate
):
    """
    Orchestrates 'target' with multiple thresholds, stores them on disk, and returns them
    If the orchestrations already exist on desk, they are not redone but simply loaded and returned
    :param target: audio
    :param save_path: Path object to save the orchestrations and check for existing orchestrations
    :return: list of solutions
    """
    orchestrations = []
    # save_path.mkdir(parents=True, exist_ok=True)
    for onset_threshold in thresholds:
        # save_name = "threshold" + str(onset_threshold)
        # save_name = save_name.replace(".", "_") + ".wav"
        # save_path = save_path / save_name
        if os.path.exists(save_path):
            # print("Found orchestration on disk")
            solution, _ = librosa.load(save_path, sr=sample_rate)
        else:
            set_config_parameter(config_path, "onsets_threshold", onset_threshold)
            solution = orchestrate(waveform)
            # sf.write(save_path, solution, samplerate=sample_rate)
        # save_path = save_path.parent
        orchestrations.append(solution)
    return orchestrations


def save_best_orchestration(orchestrations, distances, file_path):
    """
    Save the solution in 'solutions' with shortest distance in 'distances'.
    Writes a wav to 'file_path'
    :param solutions: list of solutions as numpy arrays
    :param distances: list of distances, where distances[i] is the distance from the target
    to solutions[i]
    :param file_path: full path to write the file to
    :return: None
    """
    index = distances.index(min(distances))
    best_solution = orchestrations[index]
    sf.write(file_path, best_solution, config["audio"].getint("sample_rate"))


def clean_orch_config(config_path, sol_path, analysis_db_path):
    copyfile("orch_config_template.txt", config_path)
    set_config_parameter(config_path, "sound_paths", sol_path)
    set_config_parameter(config_path, "db_files", analysis_db_path)


def compute_distances(distance_metric, target_waveform, orchestrations):
    distances = []
    for orch in orchestrations:
        distances.append(distance_metric(target_waveform, orch))
    return distances


def orch_one_target(target_file_struct, n_sources, distance_metric):

    # Full target orchestration
    sample_rate = config["audio"].getint("sample_rate")
    target_waveform = librosa.load(target_file_struct.get_path(), sr=sample_rate)
    orchestrations = orchestrate_with_threshold(
        target_waveform,
        os.path.join(target_file_struct.get_orch_folder, 'full_target.wav'),
        config["orchestration"]["thresholds"].split(", "),
        config["paths"]["config_path"],
        sample_rate,
    )
    distances = compute_distances(distance_metric, target_waveform, orchestrations)
    filename = target_file_struct.get_full_target_path()
    save_best_orchestration(orchestrations, distances, filename)

    # Divide the full orchestra into n_sources orchestras
    sub_orchestras = assign_orchestras(
        n_sources, config["orchestration"]["full_orchestra"].split(", ")
    )

    # Ground truth orchestration
    for idx, sample in enumerate(target_file_struct.get_samples_paths()):
        sample_waveform = librosa.load(sample, sr=sample_rate)
        set_config_parameter(
            config["paths"]["config_path"], "orchestra", sub_orchestras[idx]
        )
        orchestrations = orchestrate_with_threshold(
            sample_waveform,
            config["paths"]["save_path"],
            config["orchestration"]["thresholds"].split(", "),
            config["paths"]["config_path"],
            sample_rate,
        )
        distances = compute_distances(distance_metric, sample_waveform, orchestrations)
        filename = ""  # TODO
        save_best_orchestration(orchestrations, distances, filename)

    # Separated orchestration
    for separation_method in config["separation"]["methods"].split(", "):
        for idx, sample in enumerate(
            target_file_struct.get_separated_paths(separation_method)
        ):
            sample_waveform = librosa.load(sample, sr=sample_rate)
            set_config_parameter(
                config["paths"]["config_path"], "orchestra", sub_orchestras[idx]
            )
            orchestrations = orchestrate_with_threshold(
                sample_waveform,
                config["paths"]["save_path"],
                config["orchestration"]["thresholds"].split(", "),
                config["paths"]["config_path"],
                sample_rate,
            )
            distances = compute_distances(
                distance_metric, sample_waveform, orchestrations
            )
            filename = ""  # TODO
            save_best_orchestration(orchestrations, distances, filename)

    print("Done")

    # target_waveform, _ = librosa.load(target.get_mixture_path(), sr=sample_rate)
    # save_path = Path(config["paths"]["all_orchestrations_path"]) / target.name

    # # TODO: this should probably be outside (before) the function?
    # clean_orch_config(
    #     config["paths"]["config_path"],
    #     config["paths"]["sol_path"],
    #     config["paths"]["analysis_db_path"],
    # )

    # # Orchestrate full (non-separated) target
    # print("Orchestrating full target")
    # set_config_parameter(
    #     config["paths"]["config_path"],
    #     "orchestra",
    #     config["orchestration"]["full_orchestra"].split(", "),
    # )
    # full_target_orchestrations = []
    # distances = []
    # save_path = save_path / "full_orchestration"
    # orchestrations = orchestrate_with_threshold(target_waveform, save_path)
    # full_target_orchestrations = orchestrations
    # for orch in orchestrations:
    #     distance = distance_metric(target_waveform, orch)
    #     distances.append(distance)
    # full_target_distances.append(distances)
    # save_path = save_path.parent

    # # Separate target into subtargets using different separator functions
    # print("Separating target into subtargets")
    # # all_subtargets maps a model name to a list of subtargets
    # all_subtargets = {}
    # for model, separator in separation_functions.items():
    #     subtargets, _ = separator(file_struct.target, n_sources)
    #     all_subtargets[model] = subtargets

    # # orchestrate subtargets with different segmentation thresholds
    # print("Orchestrating subtargets with different thresholds")
    # orchestras = assign_orchestras(
    #     n_sources, config["orchestration"]["full_orchestra"].split(", ")
    # )
    # # orchestrated_subtargets[model][j][k] is
    # # the jth subtarget, separated via 'model', orchestrated with threshold k
    # orchestrated_subtargets = {}
    # for model in separation_functions.keys():
    #     orchestrated_subtargets[model] = [[] for _ in range(n_sources)]

    # for (model, subtargets,) in all_subtargets.items():  # for each separation algorithm
    #     save_path = save_path / model
    #     for j in range(len(subtargets)):  # for each subtarget
    #         subtarget = subtargets[j]
    #         orchestra = orchestras[j]
    #         set_config_parameter(config["paths"]["config_path"], "orchestra", orchestra)
    #         save_name = "subtarget" + str(j)
    #         save_path = save_path / save_name
    #         orchestrations = orchestrate_with_threshold(subtarget, save_path)
    #         orchestrated_subtargets[model][j] = orchestrations
    #         save_path = save_path.parent
    #     save_path = save_path.parent

    # # create all possible combinations of orchestrated subtargets and calculate distance
    # # print("Combining subtargets and calculating distance")
    # combinations = {}
    # for model, subtargets in orchestrated_subtargets.items():
    #     # for each separation algorithm
    #     combinations[model] = create_combinations(subtargets)
    #     distances = []
    #     for orch in combinations[model]:
    #         distance = distance_metric(target, orch)
    #         distances.append(distance)
    #     separated_target_distances[model].append(distances)

    # # calculate ground truth
    # print("Orchestrating ground truth")
    # # read metadata
    # with open(TARGET_METADATA_PATH, "r") as metadata_file:
    #     metadata = json.load(metadata_file)
    # metadata = metadata[target.name]
    # sources = []
    # for data in metadata:
    #     audio, _ = librosa.load(data["path"], sr=sample_rate)
    #     audio = np.pad(audio, data["padding"])
    #     sources.append(audio)
    # if len(sources) != n_sources:
    #     if len(sources) == 2:
    #         # this biases the results
    #         orchestras[0] += orchestras[2]
    #         orchestras[1] += orchestras[3]
    #         orchestras = orchestras[:2]
    #     else:
    #         raise Exception(
    #             "Code is not ready to handle {} sources".format(len(sources))
    #         )
    # ground_truth_solutions = [[] for _ in range(len(sources))]
    # save_path = save_path / "ground_truth"
    # for i in range(len(sources)):
    #     source = sources[i]
    #     orchestra = orchestras[i]
    #     set_config_parameter(config["paths"]["config_path"], "orchestra", orchestra)
    #     save_name = "source" + str(i)
    #     save_path = save_path / save_name

    #     orchestrations = orchestrate_with_threshold(source, save_path)
    #     ground_truth_solutions[i] = orchestrations
    #     save_path = save_path.parent
    # save_path = save_path.parent

    # ground_truth_combinations = create_combinations(ground_truth_solutions)
    # distances = []
    # for orch in ground_truth_combinations:
    #     distance = distance_metric(target, orch)
    #     distances.append(distance)
    # ground_truth_distances.append(distances)

    # if num_orchestrations_to_save > 0:
    #     orch_folder_path = os.path.join(
    #         config["paths"]["saved_orchestrations_path"], target.name
    #     )
    #     if not os.path.exists(orch_folder_path):
    #         os.mkdir(orch_folder_path)
    #     # save target
    #     copyfile(
    #         file_struct.target, os.path.join(orch_folder_path, target.name + ".wav")
    #     )
    #     # save best full target orchestration
    #     distances = full_target_distances[-1]
    #     name = os.path.join(orch_folder_path, "full_orchestration.wav")
    #     save_best_orchestration(full_target_orchestrations, distances, name)
    #     # save best separated orchestration
    #     for model, combos in combinations.items():
    #         distances = separated_target_distances[model][-1]
    #         name = os.path.join(orch_folder_path, model + "_orchestration.wav")
    #         save_best_orchestration(combos, distances, name)
    #     # save best ground truth orchestration
    #     distances = ground_truth_distances[-1]
    #     name = os.path.join(orch_folder_path, "ground_truth_orchestration.wav")
    #     save_best_orchestration(ground_truth_combinations, distances, name)
    #     # save distances in .txt file
    #     with open(os.path.join(orch_folder_path, "distances.txt"), "w") as f:
    #         f.write("full target distance: {}\n".format(full_target_distances[-1]))
    #         f.write("ground truth distance: {}\n".format(ground_truth_distances[-1]))
    #         for model, combos in combinations.items():
    #             f.write(
    #                 "{} distance: {}\n".format(
    #                     model, separated_target_distances[model][-1]
    #                 )
    #             )
    #     # move source separation results
    #     separations_folder = os.path.join(orch_folder_path, "separations")
    #     if not os.path.exists(separations_folder):
    #         os.mkdir(separations_folder)
    #     for model_name in config["separation"]["methods"].split(", "):
    #         # WORKAROUND (remove eventually)
    #         target.name = target.name.replace("*", "_")
    #         folder = os.path.join(SEPARATIONS_PATH, model_name, target.name)
    #         if not os.path.exists(os.path.join(separations_folder, model_name)):
    #             os.mkdir(os.path.join(separations_folder, model_name))
    #         for source in os.listdir(folder):
    #             source_path = os.path.join(folder, source)
    #             copyfile(
    #                 source_path, os.path.join(separations_folder, model_name, source),
    #             )
    #     num_orchestrations_to_save -= 1


def main(ds_path, n_sources):
    targets_paths = os.path.join(ds_path, "targets", f"{n_sources}sources")
    targets_files = librosa.util.find_files(targets_paths)
    distance_metric = frame_distance(cosine_distance)

    for fname in tqdm(targets_files):
        target_name = remove_extension(os.path.basename(fname))
        target_file_struct = TargetFileStruct(ds_path, target_name, n_sources=n_sources)
        orch_one_target(target_file_struct, n_sources, distance_metric)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("ds_path", type=str)
    parser.add_argument(
        "--n-sources",
        type=int,
        help="Number of samples to combine to get a target.",
        default=config["separation"].getint("n_sources"),
    )
    args, _ = parser.parse_known_args()
    main(args.ds_path, args.n_sources)
