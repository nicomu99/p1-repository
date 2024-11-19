import numpy as np
from sklearn.decomposition import PCA, FactorAnalysis
import matplotlib.pyplot as plt


def pca_projections_2d(data_3d):
    projections_2d = []
    for object_3d in data_3d:
        pca = PCA(n_components=2)
        projections_2d.append(pca.fit_transform(object_3d))
    return projections_2d


def varimax_projections_2d(data_3d, get_1st_and_3rd_component=False):
    projections_2d = []
    for object_3d in data_3d:
        transformer = FactorAnalysis(n_components=3, random_state=0, rotation='varimax')
        result = transformer.fit_transform(object_3d)
        if get_1st_and_3rd_component:
            projections_2d.append(result[:, [0, 2]])
        else:
            projections_2d.append(result[:, [0, 1]])
    return projections_2d


def asymmetries_x_axis(projections_2d, title, draw=True, stepsize=2, tolerance_percentage=0):
    asymmetries = []
    tolerance = stepsize / 2

    if draw:
        fig, axs = plt.subplots(len(projections_2d), 2, figsize=(8, 4 * len(projections_2d) / 3))

    for i, projection in enumerate(projections_2d):

        x_range_min, x_range_max = (min(projection[:, 0]), max(projection[:, 0]))

        if draw:
            axs[i][0].scatter(projections_2d[i][:, 0], projections_2d[i][:, 1], marker='.', alpha=0.01,
                              c='black')

        asymmetry = 0
        asymmetry_values = []

        for step in np.arange(x_range_min, x_range_max, stepsize):
            points_in_area = np.array(
                [[x, y] for x, y in projections_2d[i] if step - tolerance <= x < step + tolerance])

            if len(points_in_area) is not 0:
                if draw:
                    axs[i][0].scatter(points_in_area[:, 0], points_in_area[:, 1], marker='.', alpha=0.01,
                                      color='yellow')
                number_of_considered_values = round(
                    len(points_in_area) * tolerance_percentage) if tolerance_percentage is not 0 else 1

                if tolerance_percentage is 0 or number_of_considered_values == 0:
                    maximum = max(points_in_area[:, 1])
                else:
                    maximum = np.mean(
                        points_in_area[np.argpartition(points_in_area[:, 1], -number_of_considered_values)[
                                       -number_of_considered_values:]][:, 1])

                if tolerance_percentage is 0 or number_of_considered_values == 0:
                    minimum = min(points_in_area[:, 1])
                else:
                    minimum = np.mean(
                        points_in_area[np.argpartition(points_in_area[:, 1], number_of_considered_values)[
                                       :number_of_considered_values]][:, 1])

                if minimum == maximum:
                    asymmetry_value = maximum + minimum
                if np.sign(maximum) == 1 and np.sign(minimum) == -1:
                    asymmetry_value = abs(maximum + minimum) * 2
                if np.sign(maximum) == 1 and np.sign(minimum) == 1:
                    asymmetry_value = maximum + minimum
                if np.sign(maximum) == -1 and np.sign(minimum) == -1:
                    asymmetry_value = abs(maximum) + abs(minimum)
                if minimum > maximum:
                    print('higher min than max --> choose smaller tolerance percentage')
                    asymmetry_value = 10

                asymmetry = asymmetry + asymmetry_value

                if draw:
                    axs[i][0].scatter(step, maximum, marker='.', color='green')
                    axs[i][0].scatter(step, minimum, marker='.', color='red')

                asymmetry_values.append(asymmetry_value)
        if draw:
            axs[i][1].plot(asymmetry_values)
        asymmetries.append(asymmetry)
    if draw:
        fig.suptitle(title)
        fig.subplots_adjust(top=0.95)
    return asymmetries


def asymmetries_y_axis(projections_2d, title, draw=True, stepsize=2):
    asymmetries_2nd_PC = []
    tolerance = stepsize / 2

    if draw:
        fig, axs = plt.subplots(len(projections_2d), 2, figsize=(8, 4 * len(projections_2d) / 3))

    for i, projection in enumerate(projections_2d):

        y_range_min, y_range_max = (min(projection[:, 1]), max(projection[:, 1]))

        if draw:
            axs[i][0].scatter(projections_2d[i][:, 0], projections_2d[i][:, 1], marker='.', alpha=0.01,
                              c='black')

        asymmetry_y = 0
        asymmetry_values_y = []

        for step in np.arange(y_range_min, y_range_max, stepsize):
            points_in_area = np.array(
                [[x, y] for x, y in projections_2d[i] if y > step - tolerance and y < step + tolerance])

            # print(points_in_area)
            if draw and len(points_in_area) != 0:
                axs[i][0].scatter(points_in_area[:, 0], points_in_area[:, 1],
                                  marker='.', alpha=0.01, color='yellow')

            if len(points_in_area) != 0:
                maximum_y = max(points_in_area[:, 0])
                minimum_y = min(points_in_area[:, 0])

            if len(points_in_area) is not 0:

                if minimum_y == maximum_y:
                    asymmetry_value_y = maximum_y + minimum_y
                if np.sign(maximum_y) == 1 and np.sign(minimum_y) == -1:
                    asymmetry_value_y = abs(maximum_y + minimum_y) * 2
                if np.sign(maximum_y) == 1 and np.sign(minimum_y) == 1:
                    asymmetry_value_y = maximum_y + minimum_y
                if np.sign(maximum_y) == -1 and np.sign(minimum_y) == -1:
                    asymmetry_value_y = abs(maximum_y) + abs(minimum_y)
                asymmetry_y = asymmetry_y + asymmetry_value_y

                if draw: axs[i][0].scatter(maximum_y, step, marker='.', color='green')
                if draw: axs[i][0].scatter(minimum_y, step, marker='.', color='red')

                asymmetry_values_y.append(asymmetry_value_y)

        if draw and len(asymmetry_values_y) != 0:
            axs[i][1].plot(asymmetry_values_y)
        asymmetries_2nd_PC.append(asymmetry_y)

    if draw:
        fig.suptitle(title)
        fig.subplots_adjust(top=0.95)

    return asymmetries_2nd_PC


def min_max_asymmetries(asymmetries_of_projections_1st_axis, asymmetries_of_projections_2nd_axis):
    min_asymmetry = [min(x, y) for x, y in
                                    zip(asymmetries_of_projections_1st_axis, asymmetries_of_projections_2nd_axis)]
    max_asymmetry = [max(x, y) for x, y in
                                    zip(asymmetries_of_projections_1st_axis, asymmetries_of_projections_2nd_axis)]

    return list(zip(min_asymmetry, max_asymmetry))


def simple_rectangularity(projected_points_2d, n_bins=20):
    histograms = []
    for points in projected_points_2d:
        points = np.array(points)
        hist, _, _ = np.histogram2d(points[:, 0], points[:, 1], bins=n_bins)
        histograms.append(hist)

    ratio_of_grids_with_points_to_grids_without = []

    for histogram in histograms:
        counts = [item for sublist in histogram for item in sublist]
        if len([x for x in counts if x == 0]):
            ratio = len([x for x in counts if x != 0]) / len([x for x in counts if x == 0])
        else:
            ratio = 0
        ratio_of_grids_with_points_to_grids_without.append(ratio)

    return ratio_of_grids_with_points_to_grids_without
