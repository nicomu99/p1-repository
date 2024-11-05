import numpy as np
from sklearn.decomposition import FactorAnalysis
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

class DescriptorWrapper:
    @staticmethod
    def varimax_projection_with_scaling(point_cloud):
        fa = FactorAnalysis(n_components=2, rotation='varimax')
        transformed_pc = fa.fit_transform(point_cloud)              # Apply Factor Analysis with Varimax
        scaler = MinMaxScaler()                                     # Apply min max scaling
        return scaler.fit_transform(transformed_pc)

    @staticmethod
    def compute_evrap(point_cloud):
        # This function contains the EVRAP descriptor as defined in the thesis, i.e. a PCA with 3 components
        pca = PCA(n_components=3)
        pca.fit(point_cloud)
        return pca.explained_variance_ratio_

    def compute_samp(self, point_cloud, num_segments=2):
        scaled_pc = self.varimax_projection_with_scaling(point_cloud)

        asymmetry_values = []
        # Compute asymmetries along each axis
        for axis in range(2):
            segments = np.linspace(0, 1, num_segments + 1)          # Divide the projected data space into segments along each axis
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
        scaler = MinMaxScaler()
        normalized_asymmetry = scaler.fit_transform(np.array(asymmetry_values).reshape(-1, 1)).flatten()

        # Return the sorted tuple of asymmetry values
        return tuple(sorted(normalized_asymmetry))

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

    def compute_sirm(self, point_cloud):
        scaled_pc = self.varimax_projection_with_scaling(point_cloud)

        hull = ConvexHull(scaled_pc)
        vol_convex = hull.volume  # Area of the convex hull in 2D

        min_x, min_y = np.min(scaled_pc, axis=0)
        max_x, max_y = np.max(scaled_pc, axis=0)
        vol_bound = (max_x - min_x) * (max_y - min_y)

        return vol_convex / vol_bound if vol_bound > 0 else 0