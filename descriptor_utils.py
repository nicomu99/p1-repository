import numpy as np
from sklearn.decomposition import FactorAnalysis
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler


class DescriptorWrapper:
    @staticmethod
    def varimax_projection_with_scaling(point_cloud):
        fa = FactorAnalysis(n_components=2, random_state=0, rotation='varimax')
        transformed_pc = fa.fit_transform(point_cloud)  # Apply Factor Analysis with Varimax
        scaler = MinMaxScaler()  # Apply min max scaling
        return scaler.fit_transform(transformed_pc)

    @staticmethod
    def varimax_projection_without_scaling(point_cloud):
        fa = FactorAnalysis(n_components=2, random_state=0, rotation='varimax')
        return fa.fit_transform(point_cloud)  # Apply Factor Analysis with Varimax

    @staticmethod
    def compute_evrap(point_cloud):
        # This function contains the EVRAP descriptor as defined in the thesis, i.e. a PCA with 3 components
        pca = PCA(n_components=3)
        pca.fit(point_cloud)
        return pca.explained_variance_ratio_

    def compute_samp(self, point_cloud, num_segments=20):
        scaled_pc = self.varimax_projection_with_scaling(point_cloud)

        asymmetry_values = []
        # Compute asymmetries along each axis
        for axis in range(2):
            segments = np.linspace(0, 1,
                                   num_segments + 1)  # Divide the projected data space into segments along each axis

            asymmetry_sum = 0
            for i in range(num_segments):
                # Get points in the current segment
                in_segment = (scaled_pc[:, axis] >= segments[i]) & (scaled_pc[:, axis] < segments[i + 1])
                segment_points = scaled_pc[in_segment]

                if len(segment_points) > 0:
                    other_axis = 1 - axis
                    max_value = np.median(np.sort(segment_points[:, other_axis])[-int(0.05 * len(segment_points)):])
                    min_value = np.median(np.sort(segment_points[:, other_axis])[:int(0.05 * len(segment_points))])
                    asymmetry_sum += abs(max_value - min_value)

            asymmetry_values.append(asymmetry_sum)

        # Normalize the asymmetry values using MinMax scaling
        # scaler = MinMaxScaler()
        # normalized_asymmetry = scaler.fit_transform(np.array(asymmetry_values).reshape(-1, 1)).flatten()

        # Return the sorted tuple of asymmetry values
        return list(sorted(asymmetry_values))

    def compute_scomp(self, point_cloud, num_grid_cells=30):
        scaled_pc = self.varimax_projection_with_scaling(point_cloud)

        grid_spacex = np.linspace(0, 1, num_grid_cells)
        grid_spacey = np.linspace(0, 1, num_grid_cells)

        grid = np.zeros((num_grid_cells - 1, num_grid_cells - 1))

        # Fill the grid with presence of points
        for point in scaled_pc:
            x_idx = np.searchsorted(grid_spacex, point[0]) - 1
            y_idx = np.searchsorted(grid_spacey, point[1]) - 1
            if 0 <= x_idx < num_grid_cells - 1 and 0 <= y_idx < num_grid_cells - 1:
                grid[x_idx, y_idx] = 1

        # Calculate the area covered by the grid cells
        vol_grids = np.sum(grid)

        # Calculate the convex hull of the projections
        hull = ConvexHull(scaled_pc)
        vol_convex = hull.volume  # Area of the convex hull in 2D

        # Calculate the concavity measure
        return vol_grids / vol_convex if vol_convex > 0 else 0

    def sirm(self, point_cloud, n_bins=20):
        scaled_pc = self.varimax_projection_without_scaling(point_cloud)

        hist, _, _ = np.histogram2d(scaled_pc[:, 0], scaled_pc[:, 1], bins=n_bins)

        counts = [item for sublist in hist for item in sublist]
        if len([x for x in counts if x == 0]):
            ratio = len([x for x in counts if x != 0]) / len([x for x in counts if x == 0])
        else:
            ratio = 0

        return ratio

    def asymmetries_x_axis(self, point_cloud, stepsize=20, tolerance_percentage=0):
        scaled_pc = self.varimax_projection_with_scaling(point_cloud)
        tolerance = stepsize / 2

        x_range_min, x_range_max = (min(scaled_pc[:, 0]), max(scaled_pc[:, 0]))

        asymmetry = 0
        for step in np.arange(x_range_min, x_range_max, stepsize):
            points_in_area = np.array(
                [[x, y] for x, y in scaled_pc if step - tolerance <= x < step + tolerance])

            if len(points_in_area) != 0:
                number_of_considered_values = round(
                    len(points_in_area) * tolerance_percentage) if tolerance_percentage != 0 else 1

                if tolerance_percentage == 0 or number_of_considered_values == 0:
                    maximum = max(points_in_area[:, 1])
                else:
                    maximum = np.mean(
                        points_in_area[np.argpartition(points_in_area[:, 1], -number_of_considered_values)[
                                       -number_of_considered_values:]][:, 1])

                if tolerance_percentage == 0 or number_of_considered_values == 0:
                    minimum = min(points_in_area[:, 1])
                else:
                    minimum = np.mean(
                        points_in_area[np.argpartition(points_in_area[:, 1], number_of_considered_values)[
                                       :number_of_considered_values]][:, 1])

                asymmetry_value = 0
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

        return asymmetry

    def asymmetries_y_axis(self, point_cloud, stepsize=2):
        scaled_pc = self.varimax_projection_with_scaling(point_cloud)
        tolerance = stepsize / 2

        y_range_min, y_range_max = (min(scaled_pc[:, 1]), max(scaled_pc[:, 1]))

        asymmetry_y = 0
        for step in np.arange(y_range_min, y_range_max, stepsize):
            points_in_area = np.array(
                [[x, y] for x, y in scaled_pc if step - tolerance < y < step + tolerance])

            minimum_y = 0
            maximum_y = 0
            if len(points_in_area) != 0:
                maximum_y = max(points_in_area[:, 0])
                minimum_y = min(points_in_area[:, 0])

            if len(points_in_area) != 0:

                asymmetry_value_y = 0
                if minimum_y == maximum_y:
                    asymmetry_value_y = maximum_y + minimum_y
                if np.sign(maximum_y) == 1 and np.sign(minimum_y) == -1:
                    asymmetry_value_y = abs(maximum_y + minimum_y) * 2
                if np.sign(maximum_y) == 1 and np.sign(minimum_y) == 1:
                    asymmetry_value_y = maximum_y + minimum_y
                if np.sign(maximum_y) == -1 and np.sign(minimum_y) == -1:
                    asymmetry_value_y = abs(maximum_y) + abs(minimum_y)
                asymmetry_y = asymmetry_y + asymmetry_value_y

        return asymmetry_y

    def samp(self, point_cloud):
        asymmetry_x =  self.asymmetries_x_axis(point_cloud)
        asymmetry_y = self.asymmetries_y_axis(point_cloud)

        return [min(asymmetry_x, asymmetry_y), max(asymmetry_x, asymmetry_y)]

    def samp_on_dataset(self, point_clouds):
        asymmetries = []
        for point_cloud in point_clouds:
            asymmetries.append(self.samp(point_cloud))
        asymmetries = np.array(asymmetries)

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_asymmetries_x = scaler.fit_transform(asymmetries[:,0]).reshape(-1, 1)

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_asymmetries_y = scaler.fit_transform(asymmetries[:,1]).reshape(-1, 1)

        return self.min_max_asymmetries(scaled_asymmetries_x, scaled_asymmetries_y)

    @staticmethod
    def min_max_asymmetries(asymmetry_x, asymmetry_y):
        min_asymmetries = [min(x, y) for x, y in
                           zip(asymmetry_x,
                               asymmetry_y)]
        max_asymmetries = [max(x, y) for x, y in
                           zip(asymmetry_x,
                               asymmetry_y)]

        return np.array(list(zip(min_asymmetries, max_asymmetries)))
