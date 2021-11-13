import numpy as np

RATIO = 2
DISTANCE = 1


def lev_ratio(source, target):
    return lev(source, target, distance_or_ratio=RATIO)


def lev_distance(source, target):
    return lev(source, target, distance_or_ratio=DISTANCE)


def lev(source, target, distance_or_ratio):
    rows = len(source) + 1
    cols = len(target) + 1

    distance = np.zeros((rows, cols), dtype=int)
    distance[0, :] = np.arange(cols, dtype=int)
    distance[:, 0] = np.arange(rows, dtype=int)
    for col in range(1, cols):
        for row in range(1, rows):
            cost = 0 if source[row - 1] == target[col - 1] else distance_or_ratio
            distance[row, col] = min(
                distance[row - 1, col] + 1,
                distance[row, col - 1] + 1,
                distance[row - 1, col - 1] + cost,
            )
    Ratio = ((rows + cols + 2) - distance[row, col]) / (rows + cols + 2)
    return Ratio
