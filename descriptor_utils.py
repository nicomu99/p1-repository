import numpy as np
from sklearn.decomposition import FactorAnalysis
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import open3d as o3d
import samp as smp

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
        pca = PCA(n_components=3, random_state=0)
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
                    if len(np.sort(segment_points[:, other_axis])[-int(0.05 * len(segment_points)):]) > 0:
                        max_value = np.median(np.sort(segment_points[:, other_axis])[-int(0.05 * len(segment_points)):])
                    else:
                        max_value = 0
                    if len(np.sort(segment_points[:, other_axis])[:int(0.05 * len(segment_points))]) > 0:
                        min_value = np.median(np.sort(segment_points[:, other_axis])[:int(0.05 * len(segment_points))])
                    else:
                        min_value = 0

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

    @staticmethod
    def compute_esf(self, point_cloud, num_bins=60):
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

    def shell_model(self, point_cloud, num_bins=12):
        dist_to_center = np.sqrt(np.sum(point_cloud ** 2, axis=1))

        bins = np.linspace(0, dist_to_center.max(), num_bins + 1)

        # Compute histogram
        histogram, _ = np.histogram(dist_to_center, bins=bins, density=False)

        return histogram

    def cartesian_to_spherical(self, points):
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

    def compute_fpfh(self, point_cloud):
        o3d_pc = o3d.geometry.PointCloud()
        o3d_pc.points = o3d.utility.Vector3dVector(point_cloud)
        o3d_pc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
        fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            o3d_pc,
            o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=100),
        )
        fpfh_descriptors = np.array(fpfh.data).T
        return np.count_nonzero(fpfh_descriptors, axis=0)

    def compute_model_on_dataset(self, point_clouds, model='evrap', **kwargs):
        model_functions = {
            'evrap': self.compute_evrap,
            'sirm': self.sirm,
            'scomp': self.compute_scomp,
            'sector_model': self.sector_model,
            'shell_model': self.shell_model,
            'combined_model': self.combined_model,
            'fpfh': self.compute_fpfh,
            'samp': smp.compute_samp_on_dataset
        }

        func = model_functions[model]
        if not func:
            raise ValueError('No model function for {}'.format(model))

        descriptor = []
        for cloud in point_clouds:
            descriptor.append(func(cloud, **kwargs))
        return np.array(descriptor)
