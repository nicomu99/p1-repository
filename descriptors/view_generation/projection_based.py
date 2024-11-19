import numpy as np
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from tqdm import tqdm


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


def asymmetries_x_axis(projections_2d, n_segments=20, considered_percentage=0.05):
    normalized_projections = []
    min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
    for unnormalized_projection in projections_2d:
        normalized = min_max_scaler.fit_transform(np.array(unnormalized_projection))
        normalized_projections.append(normalized)

    asymmetries = []
    asymmetries_single_values = []

    for i, projection in enumerate(tqdm(normalized_projections)):
        asymmetry = 0
        asymmetry_values = []

        segments = np.linspace(min(projection[:, 0]), max(projection[:, 0]), n_segments, endpoint=True)

        for j, step in enumerate(segments[0:-1]):
            min_x_area = step
            max_x_area = segments[j + 1]
            points_in_area = np.array(
                [[x, y] for x, y in normalized_projections[i] if min_x_area <= x < max_x_area])

            number_of_considered_values = int(round(len(points_in_area) * considered_percentage))
            if len(points_in_area) != 0:
                if number_of_considered_values == 0:
                    maximum = max(points_in_area[:, 1])
                else:
                    maximum = np.median(
                        points_in_area[
                            np.argpartition(points_in_area[:, 1], -number_of_considered_values)[
                            -number_of_considered_values:]][:, 1])

                if number_of_considered_values == 0:
                    minimum = min(points_in_area[:, 1])
                else:
                    try:
                        minimum = np.median(
                            points_in_area[
                                np.argpartition(points_in_area[:, 1], number_of_considered_values)[
                                :number_of_considered_values]][:, 1])

                    except ValueError:
                        print('points in area ', points_in_area[:, 1])
                        print('number of considered values', number_of_considered_values)
                        pass

                asymmetry_value = abs(maximum + minimum)
                asymmetry = asymmetry + asymmetry_value

        asymmetries_single_values.append(asymmetry_values)
        asymmetries.append(asymmetry)
    return asymmetries, asymmetries_single_values


def asymmetries_y_axis(projections_2d, n_segments=20, considered_percentage=0.05):
    # normalize the points
    normalized_projections = []
    min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
    for unnormalized_projection in projections_2d:
        normalized_projections.append(
            min_max_scaler.fit_transform(np.array(unnormalized_projection)))

    asymmetries_2nd_PC = []
    asymmetries_single_values = []

    for i, projection in enumerate(tqdm(normalized_projections)):
        asymmetry_y = 0
        asymmetry_values_y = []
        segments = np.linspace(min(projection[:, 1]), max(projection[:, 1]), n_segments,
                               endpoint=True)
        tolerance = abs(segments[0] - segments[1]) / 2

        for j, step in enumerate(segments[0:-1]):
            min_y_area = step
            max_y_area = segments[j + 1]
            points_in_area = np.array(
                [[x, y] for x, y in normalized_projections[i] if min_y_area <= y < max_y_area])

            if len(points_in_area) != 0:
                number_of_considered_values = int(
                    round(len(points_in_area) * considered_percentage))
                maximum_y = np.median(
                    points_in_area[
                        np.argpartition(points_in_area[:, 0], -number_of_considered_values)
                        [-number_of_considered_values:]][:, 0])
                minimum_y = np.median(
                    points_in_area[
                        np.argpartition(points_in_area[:, 0], number_of_considered_values)
                        [:number_of_considered_values]][:, 0])

                asymmetry_value_y = abs(maximum_y + minimum_y)

                asymmetry_y = asymmetry_y + asymmetry_value_y
                asymmetry_values_y.append(asymmetry_value_y)
        asymmetries_2nd_PC.append(asymmetry_y)
        asymmetries_single_values.append(asymmetry_values_y)

    return asymmetries_2nd_PC, asymmetries_single_values


def min_max_asymmetries(asymmetries_of_projections_1st_axis, asymmetries_of_projections_2nd_axis):
    min_asymmetries = [min(x, y) for x, y in
                       zip(asymmetries_of_projections_1st_axis,
                           asymmetries_of_projections_2nd_axis)]
    max_asymmetries = [max(x, y) for x, y in
                       zip(asymmetries_of_projections_1st_axis,
                           asymmetries_of_projections_2nd_axis)]

    return list(zip(min_asymmetries, max_asymmetries))


def samp_on_point_cloud(point_cloud):
    asymmetry_x = asymmetries_x_axis(point_cloud, "", False)
    asymmetry_y = asymmetries_y_axis(point_cloud, "", )

    return [min(asymmetry_x, asymmetry_y), max(asymmetry_x, asymmetry_y)]


def samp(point_clouds):
    asymmetries = []
    for point_cloud in point_clouds:
        asymmetries.append(samp_on_point_cloud(point_cloud))
    asymmetries = np.array(asymmetries)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_asymmetries_x = scaler.fit_transform(asymmetries[:, 0]).reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_asymmetries_y = scaler.fit_transform(asymmetries[:, 1]).reshape(-1, 1)

    return min_max_asymmetries(scaled_asymmetries_x, scaled_asymmetries_y)


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
