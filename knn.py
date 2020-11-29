# https://towardsdatascience.com/machine-learning-basics-with-the-k-nearest-neighbors-algorithm-6a6e71d01761
from collections import Counter
import math

reg_data = [
    [65.75, 112.99],
    [71.52, 136.49],
    [69.40, 153.03],
    [68.22, 142.34],
    [67.79, 144.30],
    [68.70, 123.30],
    [69.80, 141.49],
    [70.01, 136.46],
    [67.90, 112.37],
    [66.49, 127.45],
]

reg_query = [60]

clf_data = [
    [22, 1],
    [23, 1],
    [21, 1],
    [18, 1],
    [19, 1],
    [25, 0],
    [27, 0],
    [29, 0],
    [31, 0],
    [45, 0],
]

clf_query = [33]


def mean(labels):
    return sum(labels)/len(labels)


def mode(labels):
    return Counter(labels).most_common(1)[0][0]


def euclidean_distance(point1, point2):
    sum_squared_distance = 0
    for i in range(len(point1)):
        sum_squared_distance += math.pow(point1[i] - point2[i], 2)

    return math.sqrt(sum_squared_distance)


def knn(repo, query, k, distance_fn, choice_fn):
    neighbor_distances_and_indices = []

    for idx, sample in enumerate(repo):
        distance = distance_fn(sample[:-1], query)
        neighbor_distances_and_indices.append((distance, idx))

    sorted_neighbor_distances_and_indices = sorted(
        neighbor_distances_and_indices)
    k_sorted_neighbor_distances_and_indices = sorted_neighbor_distances_and_indices[:k]
    k_nearest_labels = [repo[i][-1]
                        for x, i in k_sorted_neighbor_distances_and_indices]

    return k_sorted_neighbor_distances_and_indices, k_nearest_labels


reg_k_nearest_neighbors, reg_prediction = knn(
    reg_data, reg_query, k=3, distance_fn=euclidean_distance, choice_fn=mean
)

clf_k_nearest_neighbors, clf_prediction = knn(
    clf_data, clf_query, k=3, distance_fn=euclidean_distance, choice_fn=mode
)
