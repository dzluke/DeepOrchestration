import json

from pipeline import thresholds, mean, RESULTS_PATH

if __name__ == "__main__":
    with open(RESULTS_PATH, 'r') as results_file:
        results = json.load(results_file)

    num_completed = results['num_completed']
    full_target_distances = results['full_target_distances']
    separated_target_distances = results['separated_target_distances']
    ground_truth_distances = results['ground_truth_distances']

    min_full_target_distances = list(map(min, full_target_distances))
    print("Average of minimum full target distances:", mean(min_full_target_distances))

    for i in range(len(thresholds)):
        threshold_distance = [x[i] for x in full_target_distances]
        avg_threshold_distance = mean(threshold_distance)
        print("Average full target distance for threshold {}: {}".format(thresholds[i], avg_threshold_distance))

    print("Average of minimum separated distances")
    for model, distances in separated_target_distances.items():
        min_distances = list(map(min, distances))
        avg_min_distances = mean(min_distances)
        print("\t{} : {}".format(model, avg_min_distances))

    min_ground_truth_distances = list(map(min, ground_truth_distances))
    print("Average of minimum ground truth distances:", mean(min_ground_truth_distances))
