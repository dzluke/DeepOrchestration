import json

from pipeline import thresholds, mean, RESULTS_PATH

if __name__ == "__main__":
    with open(RESULTS_PATH, 'r') as results_file:
        results = json.load(results_file)

    num_completed = results['num_completed']
    full_target_distances = results['full_target_distances']
    separated_target_distances = results['separated_target_distances']
    ground_truth_distances = results['ground_truth_distances']

    avg_full_target_distance = [mean(x) for x in full_target_distances]
    print("Average full target distance:", mean(avg_full_target_distance))
    print("Average separated distance:")
    for model, distances in separated_target_distances.items():
        print("\t{} : {}".format(model, mean([mean(x) for x in distances])))
    avg_ground_truth_distance = [mean(x) for x in ground_truth_distances]
    print("Average ground truth distance:", mean(avg_ground_truth_distance))

    print("---------------------------")

    min_full_target_distances = list(map(min, full_target_distances))
    print("Average of minimum full target distances:", mean(min_full_target_distances))

    print("Average of minimum separated distances")
    for model, distances in separated_target_distances.items():
        min_distances = list(map(min, distances))
        avg_min_distances = mean(min_distances)
        print("\t{} : {}".format(model, avg_min_distances))

    min_ground_truth_distances = list(map(min, ground_truth_distances))
    print("Average of minimum ground truth distances:", mean(min_ground_truth_distances))

    print("-----------------------------------")

    for i in range(len(thresholds)):
        threshold_distance = [x[i] for x in full_target_distances]
        avg_threshold_distance = mean(threshold_distance)
        print("Average full target distance for threshold {}: {}".format(thresholds[i], avg_threshold_distance))

    threshold_indices = [0, 40, 80]
    print("Average distances across thresholds per method")
    for model, distances in separated_target_distances.items():
        print("\t" + model)
        for i in range(len(thresholds)):
            index = threshold_indices[i]
            threshold_distance = [x[index] for x in distances]
            avg_threshold_distance = mean(threshold_distance)
            print("\t\tAverage distance for threshold {}: {}".format(thresholds[i], avg_threshold_distance))

