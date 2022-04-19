from configparser import ConfigParser
import os
import random
import subprocess
from shutil import copyfile
import json

import librosa
import soundfile as sf

from pipeline import frame_distance, spectral_distance
from utils import set_config_parameter

# Get configuration
config = ConfigParser(inline_comment_prefixes="#")
config.read("config.ini")


def orchestrate(target, sample_rate, config_path):
    """
    Orchestrate 'target' using Orchidea
    :param target: sound file as numpy array, shape (len,)
    :return: orchestrated solution as numpy array, shape (len,)
    """
    sf.write('target.wav', target, sample_rate)
    cmd = ["./orchestrate", "target.wav", config_path]
    subprocess.run(cmd, stdout=subprocess.DEVNULL)  # this suppresses output
    solution, _ = librosa.load('connection.wav', sr=sample_rate)
    return solution


def assign_orchestras(num_orchestras, full_orchestra):
    """
    Randomly divide the full orchestra into 'num_orchestras' sub-orchestras
    :param num_orchestras: the number of orchestras you want the full orchestra divided into
    :return orchestras: a nested list of orchestras where len(orchestras) == num_orchestras
    """
    orchestras = []  # nested list of orchestras, orchestras[i] is the orchestra for segment i
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


def orchestrate_with_threshold(target, save_path, thresholds,
                               config_path, sample_rate):
    """
    Orchestrates 'target' with multiple thresholds, stores them on disk, and returns them
    If the orchestrations already exist on desk, they are not redone but simply loaded and returned
    :param target: audio
    :param save_path: Path object to save the orchestrations and check for existing orchestrations
    :return: list of solutions
    """
    solutions = []
    save_path.mkdir(parents=True, exist_ok=True)
    for onset_threshold in thresholds:
        save_name = "threshold" + str(onset_threshold)
        save_name = save_name.replace(".", "_") + ".wav"
        save_path = save_path / save_name
        if os.path.exists(save_path):
            # print("Found orchestration on disk")
            solution, _ = librosa.load(save_path, sr=sample_rate)
        else:
            set_config_parameter(config_path, 'onsets_threshold', onset_threshold)
            solution = orchestrate(target)
            sf.write(save_path, solution, samplerate=sample_rate)
        save_path = save_path.parent
        solutions.append(solution)
    return solutions


def clean_orch_config(config_path, sol_path, analysis_db_path):
    copyfile('orch_config_template.txt', config_path)
    set_config_parameter(config_path, 'sound_paths', sol_path)
    set_config_parameter(config_path, 'db_files', analysis_db_path)


if __name__ == "__main__":
    distance_metric = frame_distance(spectral_distance)  # the distance metric to be used to evaluate solutions

    num_completed = 0
    # full_target_distances is a nested list of distances of orchestrating full targets without separation
    # full_target_distances[i][k] is the distance between target i and the orchestration of target i with threshold k
    full_target_distances = []
    # distances for separated targets
    # separated_target_distances['model'][i] is a list of length 81 of the distances for all possible combinations (3^4)
    # of target i after being separated by 'model', orchestrated, then combined
    separated_target_distances = {}
    for model in config['separation']['methods'].split(", "):
        separated_target_distances[model] = []
    # ground_truth_distances[i] is a list of length 81, it is the distances for all possible combinations of
    # orchestrations of the ground truth source subtargets
    ground_truth_distances = []

    # a dictionary that represents the current state of the pipeline, including how many targets have
    # been completed, and the distances.
    # this is stored/loaded as a json so the process can be stopped and restarted
    results = {'num_completed': num_completed,
               'full_target_distances': full_target_distances,
               'separated_target_distances': separated_target_distances,
               'ground_truth_distances': ground_truth_distances}

    # load saved results from json, or create json if none exists
    try:
        results_file = open(RESULTS_PATH, 'r')
        results = json.load(results_file)
        if len(results) != 0:  # if it's not an empty json
            num_completed = results['num_completed']
            full_target_distances = results['full_target_distances']
            separated_target_distances = results['separated_target_distances']
            ground_truth_distances = results['ground_truth_distances']
            print("Backing up results.json to results_backup.json")
            copyfile(RESULTS_PATH, 'results_backup.json')
    except FileNotFoundError:
        results_file = open(RESULTS_PATH, 'w')
    results_file.close()

    targets = librosa.util.find_files(TARGETS_PATH)
    print("Database contains {} targets".format(len(targets)))

    if num_orchestrations_to_save > 0 and not os.path.exists(SAVED_ORCHESTRATIONS_PATH):
        os.mkdir(SAVED_ORCHESTRATIONS_PATH)

    while num_completed < len(targets):
        target_path = targets[num_completed]
        target, _ = librosa.load(target_path, sr=SAMPLING_RATE)
        target_name = os.path.basename(target_path)
        target_name = os.path.splitext(target_name)[0]
        save_path = Path(ALL_ORCHESTRATIONS_PATH) / target_name

        print("Target:", target_name)
        print("num completed:", num_completed)

        clean_orch_config(CONFIG_PATH, SOL_PATH, ANALYSIS_DB_PATH)

        # orchestrate full (non-separated) target with Orchidea
        print("Orchestrating full target")
        full_target_distance = 0
        set_config_parameter(CONFIG_PATH, 'orchestra', FULL_ORCHESTRA)
        full_target_solutions = []
        distances = []
        save_path = save_path / "full_orchestration"

        solutions = orchestrate_with_threshold(target, save_path)
        full_target_solutions = solutions
        for solution in solutions:
            distance = distance_metric(target, solution)
            distances.append(distance)
        full_target_distances.append(distances)
        save_path = save_path.parent

        # separate target into subtargets using different separator functions
        print("Separating target into subtargets")
        # all_subtargets maps a model name to a list of subtargets
        all_subtargets = {}
        for model, separator in separation_functions.items():
            subtargets, sr = separator(target_path, NUM_SUBTARGETS)
            all_subtargets[model] = subtargets

        # orchestrate subtargets with different segmentation thresholds
        print("Orchestrating subtargets with different thresholds")
        orchestras = assign_orchestras(NUM_SUBTARGETS, FULL_ORCHESTRA)
        # orchestrated_subtargets[model][j][k] is
        # the jth subtarget, separated via 'model', orchestrated with threshold k
        orchestrated_subtargets = {}
        for model in separation_functions.keys():
            orchestrated_subtargets[model] = [[] for _ in range(NUM_SUBTARGETS)]

        for model, subtargets in all_subtargets.items():  # for each separation algorithm
            save_path = save_path / model
            for j in range(len(subtargets)):  # for each subtarget
                subtarget = subtargets[j]
                orchestra = orchestras[j]
                set_config_parameter(CONFIG_PATH, 'orchestra', orchestra)
                save_name = "subtarget" + str(j)
                save_path = save_path / save_name
                solutions = orchestrate_with_threshold(subtarget, save_path)
                orchestrated_subtargets[model][j] = solutions
                save_path = save_path.parent
            save_path = save_path.parent


        # create all possible combinations of orchestrated subtargets and calculate distance
        # print("Combining subtargets and calculating distance")
        combinations = {}
        for model, subtargets in orchestrated_subtargets.items():
            # for each separation algorithm
            combinations[model] = create_combinations(subtargets)
            distances = []
            for solution in combinations[model]:
                distance = distance_metric(target, solution)
                distances.append(distance)
            separated_target_distances[model].append(distances)

        # calculate ground truth
        print("Orchestrating ground truth")
        # read metadata
        with open(TARGET_METADATA_PATH, 'r') as metadata_file:
            metadata = json.load(metadata_file)
        metadata = metadata[target_name]
        sources = []
        for data in metadata:
            audio, _ = librosa.load(data['path'], sr=SAMPLING_RATE)
            audio = np.pad(audio, data['padding'])
            sources.append(audio)
        if len(sources) != NUM_SUBTARGETS:
            if len(sources) == 2:
                # this biases the results
                orchestras[0] += orchestras[2]
                orchestras[1] += orchestras[3]
                orchestras = orchestras[:2]
            else:
                raise Exception("Code is not ready to handle {} sources".format(len(sources)))
        ground_truth_solutions = [[] for _ in range(len(sources))]
        save_path = save_path / "ground_truth"
        for i in range(len(sources)):
            source = sources[i]
            orchestra = orchestras[i]
            set_config_parameter(CONFIG_PATH, 'orchestra', orchestra)
            save_name = "source" + str(i)
            save_path = save_path / save_name

            solutions = orchestrate_with_threshold(source, save_path)
            ground_truth_solutions[i] = solutions
            save_path = save_path.parent
        save_path = save_path.parent

        ground_truth_combinations = create_combinations(ground_truth_solutions)
        distances = []
        for solution in ground_truth_combinations:
            distance = distance_metric(target, solution)
            distances.append(distance)
        ground_truth_distances.append(distances)

        if num_orchestrations_to_save > 0:
            orch_folder_path = os.path.join(SAVED_ORCHESTRATIONS_PATH, target_name)
            if not os.path.exists(orch_folder_path):
                os.mkdir(orch_folder_path)
            # save target
            copyfile(target_path, os.path.join(orch_folder_path, target_name + '.wav'))
            # save best full target orchestration
            distances = full_target_distances[-1]
            name = os.path.join(orch_folder_path, "full_orchestration.wav")
            save_best_orchestration(full_target_solutions, distances, name)
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
            with open(os.path.join(orch_folder_path, 'distances.txt'), 'w') as f:
                f.write("full target distance: {}\n".format(full_target_distances[-1]))
                f.write("ground truth distance: {}\n".format(ground_truth_distances[-1]))
                for model, combos in combinations.items():
                    f.write("{} distance: {}\n".format(model, separated_target_distances[model][-1]))
            # move source separation results
            separations_folder = os.path.join(orch_folder_path, 'separations')
            if not os.path.exists(separations_folder):
                os.mkdir(separations_folder)
            for model_name in separation_models:
                # WORKAROUND (remove eventually)
                target_name = target_name.replace("*", "_")
                folder = os.path.join(SEPARATIONS_PATH, model_name, target_name)
                if not os.path.exists(os.path.join(separations_folder, model_name)):
                    os.mkdir(os.path.join(separations_folder, model_name))
                for source in os.listdir(folder):
                    source_path = os.path.join(folder, source)
                    copyfile(source_path, os.path.join(separations_folder, model_name, source))
            num_orchestrations_to_save -= 1

        num_completed += 1
        # save results to json
        results['num_completed'] = num_completed
        results['full_target_distances'] = full_target_distances
        results['separated_target_distances'] = separated_target_distances
        results['ground_truth_distances'] = ground_truth_distances
        with open(RESULTS_PATH, 'w') as f:
            json.dump(results, f, indent='\t')

        # remove temp files created during separation
        # for model_name in separation_models:
        #     path = os.path.join(SEPARATIONS_PATH, model_name, target_name)
        #     if os.path.exists(path):
        #         remove_directory(path)

    # remove files created during pipeline
    for file in ['target.wav', 'segments.txt']:
        if os.path.exists(file):
            os.remove(file)
