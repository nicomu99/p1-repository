import numpy as np
from sklearn.decomposition import FactorAnalysis
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import open3d as o3d
import warnings
warnings.simplefilter('error', RuntimeWarning)

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
    def varimax_projection_with_scaling_3d(point_cloud):
        fa = FactorAnalysis(n_components=3, random_state=0, rotation='varimax')
        transformed_pc = fa.fit_transform(point_cloud)  # Apply Factor Analysis with Varimax
        scaler = MinMaxScaler()  # Apply min max scaling
        return scaler.fit_transform(transformed_pc)

    @staticmethod
    def varimax_projection_without_scaling_3d(point_cloud):
        fa = FactorAnalysis(n_components=3, random_state=0, rotation='varimax')
        return fa.fit_transform(point_cloud)  # Apply Factor Analysis with Varimax

    @staticmethod
    def compute_evrap(point_cloud):
        # This function contains the EVRAP descriptor as defined in the thesis, i.e. a PCA with 3 components
        pca = PCA(n_components=3, random_state=0)
        pca.fit(point_cloud)
        return pca.explained_variance_ratio_

    def compute_scomp(self, point_cloud, num_grid_cells=30):
        scaled_pc = self.varimax_projection_without_scaling(point_cloud)

        grid_spacex = np.linspace(np.min(scaled_pc[:, 0]), np.max(scaled_pc[:, 0]), num_grid_cells)
        grid_spacey = np.linspace(np.min(scaled_pc[:, 1]), np.max(scaled_pc[:, 1]), num_grid_cells)

        grid = np.zeros((num_grid_cells - 1, num_grid_cells - 1))
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
        return np.array([vol_grids / vol_convex if vol_convex > 0 else 0])

    def compute_scomp_3d(self, point_cloud, num_grid_cells=30):
        scaled_pc = self.varimax_projection_without_scaling_3d(point_cloud)

        descriptor = []
        for i, (x, y) in enumerate([(0, 1), (0, 2), (1, 2)]):
            projection_2d = scaled_pc[:, [x, y]]

            grid_spacex = np.linspace(np.min(projection_2d[:, 0]), np.max(projection_2d[:, 0]), num_grid_cells)
            grid_spacey = np.linspace(np.min(projection_2d[:, 1]), np.max(projection_2d[:, 1]), num_grid_cells)

            grid = np.zeros((num_grid_cells - 1, num_grid_cells - 1))
            for point in projection_2d:
                x_idx = np.searchsorted(grid_spacex, point[0]) - 1
                y_idx = np.searchsorted(grid_spacey, point[1]) - 1
                if 0 <= x_idx < num_grid_cells - 1 and 0 <= y_idx < num_grid_cells - 1:
                    grid[x_idx, y_idx] = 1

            vol_grids = np.sum(grid)

            hull = ConvexHull(projection_2d)
            vol_convex = hull.volume

            descriptor.append(vol_grids / vol_convex if vol_convex > 0 else 0)

        return np.array(descriptor)

    def sirm(self, point_cloud):
        scaled_pc = self.varimax_projection_without_scaling(point_cloud)

        min_x, max_x = np.min(scaled_pc[:, 0]), np.max(scaled_pc[:, 0])
        min_y, max_y = np.min(scaled_pc[:, 1]), np.max(scaled_pc[:, 1])

        area = (max_x - min_x) * (max_y - min_y)

        hull = ConvexHull(scaled_pc)
        vol_convex = hull.volume

        return np.array([vol_convex / area if area > 0 else 0])

    def sirm_3d(self, point_cloud):
        scaled_pc = self.varimax_projection_without_scaling_3d(point_cloud)

        descriptor = []
        for i, (x, y) in enumerate([(0, 1), (0, 2), (1, 2)]):
            projection_2d = scaled_pc[:, [x, y]]
            min_x, max_x = np.min(projection_2d[:, 0]), np.max(projection_2d[:, 0])
            min_y, max_y = np.min(projection_2d[:, 1]), np.max(projection_2d[:, 1])

            area = (max_x - min_x) * (max_y - min_y)

            hull = ConvexHull(scaled_pc)
            vol_convex = hull.volume

            descriptor.append(vol_convex / area if area > 0 else 0)

        return np.array(descriptor)

    @staticmethod
    def compute_esf(point_cloud, num_bins=60):
        # Convert point cloud to numpy array
        points = np.asarray(point_cloud.points)

        # Compute pairwise distances
        num_points = points.shape[0]
        distances = np.linalg.norm(points[:, np.newaxis, :] - points[np.newaxis, :, :], axis=2)
        distances = distances[np.triu_indices(num_points, k=1)]  # Upper triangular (excluding diagonal)

        # 1. Histogram of distances
        hist_d, _ = np.histogram(distances, bins=num_bins, range=(0, distances.max()), density=True)

        # 2. Histogram of angles
        angles = []
        for i in range(num_points):
            for j in range(i + 1, num_points):
                for k in range(j + 1, num_points):
                    vec_ij = points[j] - points[i]
                    vec_ik = points[k] - points[i]
                    cosine_angle = np.dot(vec_ij, vec_ik) / (np.linalg.norm(vec_ij) * np.linalg.norm(vec_ik) + 1e-8)
                    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
                    angles.append(angle)
        angles = np.array(angles)
        hist_a, _ = np.histogram(angles, bins=num_bins, range=(0, np.pi), density=True)

        # 3. Histogram of distance ratios
        distance_ratios = []
        for i in range(num_points):
            for j in range(i + 1, num_points):
                for k in range(j + 1, num_points):
                    d_ij = np.linalg.norm(points[i] - points[j])
                    d_ik = np.linalg.norm(points[i] - points[k])
                    ratio = d_ij / (d_ik + 1e-8)
                    distance_ratios.append(ratio)
        distance_ratios = np.array(distance_ratios)
        hist_t, _ = np.histogram(distance_ratios, bins=num_bins, range=(0, distance_ratios.max()), density=True)

        # Concatenate histograms to form the ESF descriptor
        esf_descriptor = np.concatenate([hist_d, hist_a, hist_t])

        return esf_descriptor

    @staticmethod
    def shell_model(point_cloud, num_bins=12):
        dist_to_center = np.sqrt(np.sum(point_cloud ** 2, axis=1))

        bins = np.linspace(0, dist_to_center.max(), num_bins + 1)

        # Compute histogram
        histogram, _ = np.histogram(dist_to_center, bins=bins, density=False)

        return histogram

    @staticmethod
    def cartesian_to_spherical(points):
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        theta = np.arccos(z / np.maximum(r, 1e-8))  # In case of division by 0
        phi = np.arctan2(y, x)

        return r, theta, phi

    def get_sector_indices(self, point_cloud, num_sectors=12):
        r, _, phi = self.cartesian_to_spherical(point_cloud)
        phi = np.mod(phi, 2 * np.pi)
        phi_bins = np.linspace(0, 2 * np.pi, num_sectors + 1)
        phi_indices = np.digitize(phi, bins=phi_bins) - 1

        return phi_indices

    def sector_model(self, point_cloud, num_sectors=12):
        phi_indices = self.get_sector_indices(point_cloud, num_sectors)
        return np.bincount(phi_indices, minlength=num_sectors)

    def combined_model(self, point_cloud, num_sectors=12, num_bins=6):
        phi_indices = self.get_sector_indices(point_cloud, num_sectors)

        histograms = []
        for index in range(num_sectors):
            sector_points = point_cloud[phi_indices == index]
            if sector_points.shape[0] == 0:
                histograms.append(np.zeros(num_bins))
                continue

            # Get all points in this area
            histograms.append(self.shell_model(sector_points, num_bins=num_bins))

        return np.concatenate(histograms)

    @staticmethod
    def compute_fpfh(point_cloud):
        o3d_pc = o3d.geometry.PointCloud()
        o3d_pc.points = o3d.utility.Vector3dVector(point_cloud)
        o3d_pc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
        fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            o3d_pc,
            o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=100),
        )
        fpfh_descriptors = np.array(fpfh.data).T
        return np.count_nonzero(fpfh_descriptors, axis=0)

    def compute_samp(self, point_cloud, n_segments=20, sampling_percentage=0.05):
        projection = self.varimax_projection_with_scaling(point_cloud)
        min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
        projection = min_max_scaler.fit_transform(projection)

        samp_descriptor = []
        segment_boundary = np.linspace(-1, 1, n_segments + 1)
        for axis_index, axis in enumerate(["x", "y"]):
            segmentation_axis = projection[:, axis_index]

            asymmetries = []
            for segment in range(n_segments):
                lower_boundary = segment_boundary[segment]
                upper_boundary = segment_boundary[segment + 1]
                mask = (segmentation_axis >= lower_boundary) & (segmentation_axis < upper_boundary)

                sorted_considered_values = np.sort(projection[mask, int(not axis_index)])
                if len(sorted_considered_values) <= 1:
                    # If 0 we can not compute a difference, if 1 we will subtract the same value form itself
                    asymmetries.append(0)
                    continue

                outlier_count = max(1, int(np.ceil(len(sorted_considered_values) * sampling_percentage)))  # At least 1 element
                top_median = np.median(sorted_considered_values[-outlier_count:])
                bottom_median = np.median(sorted_considered_values[:outlier_count])
                asymmetries.append(abs(top_median - bottom_median))
            samp_descriptor.append(np.sum(asymmetries))

        return -np.sort(-np.array(samp_descriptor))

    def compute_samp_3d(self, point_cloud, n_segments=20, sampling_percentage=0.05, absolute=True):
        projection = self.varimax_projection_with_scaling_3d(point_cloud)

        samp_3d = []
        segment_boundary = np.linspace(-1, 1, n_segments + 1)
        for x, y in [(0, 1), (0, 2), (1, 2)]:
            projection_2d = projection[:, [x, y]]

            samp_descriptor = []
            for axis_index, axis in enumerate(["x", "y"]):
                segmentation_axis = projection_2d[:, axis_index]

                asymmetries = []
                for segment in range(n_segments):
                    lower_boundary = segment_boundary[segment]
                    upper_boundary = segment_boundary[segment + 1]
                    mask = (segmentation_axis >= lower_boundary) & (segmentation_axis < upper_boundary)

                    sorted_considered_values = np.sort(projection_2d[mask, int(not axis_index)])
                    if len(sorted_considered_values) <= 1:
                        # If 0 we can not compute a difference, if 1 we will subtract the same value form itself
                        asymmetries.append(0)
                        continue

                    outlier_count = max(1, int(np.ceil(len(sorted_considered_values) * sampling_percentage)))  # At least 1 element
                    top_median = np.median(sorted_considered_values[-outlier_count:])
                    bottom_median = np.median(sorted_considered_values[:outlier_count])
                    if absolute:
                        asymmetries.append(abs(top_median - bottom_median))
                    else:
                        asymmetries.append(top_median - bottom_median)
                samp_descriptor.append(np.sum(asymmetries))

            desc = np.sort(samp_descriptor)[::-1]
            samp_3d.extend(desc[:2])
        return np.array(samp_3d)

    def compute_samp_3d_no_abs(self, point_cloud, n_segments=20, sampling_percentage=0.05):
        return self.compute_samp_3d(point_cloud, n_segments=n_segments, sampling_percentage=sampling_percentage, absolute=False)

    # Normalize each axis independently
    @staticmethod
    def normalize_per_axis(data):
        normalized_data = np.empty_like(data, dtype=float)
        for axis in range(data.shape[1]):  # Iterate over columns (axes)
            axis_min = np.min(data[:, axis])
            axis_max = np.max(data[:, axis])
            normalized_data[:, axis] = (data[:, axis] - axis_min) / (axis_max - axis_min)
        return np.array(normalized_data)

    def compute_model_on_dataset(self, point_clouds, model='evrap', **kwargs):
        model_functions = {
            'evrap': self.compute_evrap,
            'sirm': self.sirm,
            'scomp': self.compute_scomp,
            'sector_model': self.sector_model,
            'shell_model': self.shell_model,
            'combined_model': self.combined_model,
            'pfh': self.compute_fpfh,
            'samp': self.compute_samp,
            'samp_3d': self.compute_samp_3d,
            'scomp_3d': self.compute_scomp_3d,
            'sirm_3d': self.sirm_3d,
            'samp_3d_no_abs': self.compute_samp_3d_no_abs
        }

        func = model_functions[model]
        if not func:
            raise ValueError('No model function for {}'.format(model))

        descriptor = []
        for cloud in point_clouds:
            descriptor.append(func(cloud, **kwargs))
        descriptor = np.array(descriptor)

        if model == 'samp' or model == 'samp_3d' or model == 'samp_3d_no_abs':
            descriptor = self.normalize_per_axis(descriptor)

        return descriptor