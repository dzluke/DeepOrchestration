import json
from statistics import median

from pipeline import thresholds, mean, RESULTS_PATH


def aggregate_data(f1, f2, name):
    avg_full_target_distance = [f1(x) for x in full_target_distances]
    print(name, "full target distance:", f2(avg_full_target_distance))
    print(name, "separated distance:")
    for model, distances in separated_target_distances.items():
        print("\t{} : {}".format(model, f2([f1(x) for x in distances])))
    avg_ground_truth_distance = [f1(x) for x in ground_truth_distances]
    print(name, "ground truth distance:", f2(avg_ground_truth_distance))
    print("---------------------------")


if __name__ == "__main__":
    with open(RESULTS_PATH, 'r') as results_file:
        results = json.load(results_file)

    num_completed = results['num_completed']
    full_target_distances = results['full_target_distances']
    separated_target_distances = results['separated_target_distances']
    ground_truth_distances = results['ground_truth_distances']

    # aggregate_data(lambda x: x[0], median, 'Median')
    aggregate_data(lambda x: x[0], mean, 'Mean')

    x = 0  # num times separate is better than full
    y = 0  # num times ground truth is better than separate
    z = 0  # num times ground truth is better than full

    ground_increase = 0
    for i in range(num_completed):
        full = full_target_distances[i][0]
        sep = []
        for model, distances in separated_target_distances.items():
            sep.append(distances[i][0])
        sep = min(sep)
        ground = ground_truth_distances[i][0]

        if sep < full:
            x += 1
        if ground < sep:
            y += 1
        if ground < full:
            z += 1

    print("Separate is better than full:", x / num_completed * 100, "%")
    print("Ground is better than separate:", y / num_completed * 100, "%")
    print("Ground is better than full:", z / num_completed * 100, "%")

    if len(thresholds) > 1:
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

