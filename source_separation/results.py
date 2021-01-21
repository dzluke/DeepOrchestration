import librosa
import json

from pipeline import mean, num_separation_functions, TARGETS_PATH, RESULTS_PATH

if __name__ == "__main__":
    targets = librosa.util.find_files(TARGETS_PATH)
    with open(RESULTS_PATH, 'r') as results_file:
        results = json.load(results_file)

    num_completed = results['num_completed']
    full_target_distances = results['full_target_distances']
    separated_target_distances = results['separated_target_distances']

    assert len(targets) == num_completed
    assert len(targets) == len(full_target_distances)
    assert len(full_target_distances) == len(separated_target_distances)

    for lst in separated_target_distances:
        assert len(lst) == num_separation_functions

    avg_separated_distances = [mean(x) for x in separated_target_distances]

    # a list of the avg distance for each separation function
    separation_function_distances = [0 for _ in range(num_separation_functions)]
    for i in range(num_separation_functions):
        for distances in separated_target_distances:
            separation_function_distances[i] += distances[i]
    separation_function_distances = [x / num_completed for x in separation_function_distances]

    for i in range(len(targets)):
        print("Target:", targets[i].split('/')[-1])
        print("Full target distance:", full_target_distances[i])
        print("Avg separated target distance:", avg_separated_distances[i])
        print("Ratio:", full_target_distances[i] / avg_separated_distances[i])
        print("--------------------------")

    print("\n")
    print("Avg full target distances:", mean(full_target_distances))
    print("Avg separated target distances:", mean(avg_separated_distances))
    print("Ratio of averages:", mean(full_target_distances) / mean(avg_separated_distances))

    print("\n")
    print("Avg distance for each separation function")
    print(separation_function_distances)


