from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tqdm import tqdm


# Written by Lisa Krombholz

def varimax_projections_real_2d(data_3d):
    projections_2d = []
    for object_3d in data_3d:
        transformer = FactorAnalysis(n_components=2, random_state=0, rotation='varimax')
        result = transformer.fit_transform(object_3d)
        projections_2d.append(result[:, [0, 1]])
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


def get_min_max_varimax(pointclouds, n_segments=20, get_1st_and_3rd_component=False):
    if get_1st_and_3rd_component:
        projections = varimax_projections_2d(pointclouds, get_1st_and_3rd_component=True)
    else:
        projections = varimax_projections_2d(pointclouds)
    asymmetries_x, _ = asymmetries_x_axis(projections, n_segments=n_segments)
    asymmetries_y, _ = asymmetries_y_axis(projections, n_segments=n_segments)
    return min_max_asymmetries(asymmetries_x, asymmetries_y)


def get_min_max_varimax_normed_scaled(pointclouds, n_segments=20, get_1st_and_3rd_component=False):
    projections = varimax_projections_real_2d(pointclouds)

    asymmetries_x, _ = asymmetries_x_axis(projections, n_segments=n_segments, normalize=True)
    asymmetries_y, _ = asymmetries_y_axis(projections, n_segments=n_segments, normalize=True)

    scaler_x = MinMaxScaler(feature_range=(0, 1))
    scaled_x = np.array(
        scaler_x.fit_transform(np.array(asymmetries_x).reshape(-1, 1)))

    scaler_y = MinMaxScaler(feature_range=(0, 1))
    scaled_y = np.array(
        scaler_y.fit_transform(np.array(asymmetries_y).reshape(-1, 1)))

    return min_max_asymmetries(scaled_x, scaled_y)


def remove_outliers(data, max_number_std=2):
    mean = np.mean(data)
    std_dev = np.std(data)
    zero_based = abs(data - mean)
    return data[zero_based < max_number_std * std_dev]
