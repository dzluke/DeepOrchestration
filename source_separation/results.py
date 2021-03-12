import json
from statistics import median, stdev
import matplotlib.pyplot as plt

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

    y = []
    e = []
    x_labels = []

    # order: Open-Unmix, Demucs, TDCN, NMF, TDCN++, Full, Ground

    def make_plot_data(distances, label):
        distances = [x[0] for x in distances]
        y.append(mean(distances))
        e.append(stdev(distances))
        x_labels.append(label)

    make_plot_data(separated_target_distances['OpenUnmix'], 'Open-Unmix')
    make_plot_data(separated_target_distances['Demucs'], 'Demucs')
    make_plot_data(separated_target_distances['TDCNN'], 'TDCN')
    make_plot_data(separated_target_distances['NMF'], 'NMF')
    make_plot_data(separated_target_distances['TDCNN++'], 'TDCN++')
    make_plot_data(full_target_distances, 'Full target')
    make_plot_data(ground_truth_distances, 'Ground truth')

    x = list(range(len(y)))
    plt.title("Average distance between target and solution with standard deviation")
    plt.errorbar(x, y, e, linestyle='None', marker='.', markersize=10, capsize=3, color='black')
    plt.xticks(x, x_labels, rotation=15)
    plt.xlabel("Orchestration type")
    plt.ylabel("Distance")
    plt.ylim(0, 50)
    plt.tight_layout()
    plt.show()

    aggregate_data(lambda x: x[0], median, 'Median')
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

