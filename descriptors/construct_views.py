from tqdm import tqdm

# from utils.file_utils import o3d_meshes, read_pcd_pointclouds_poisson
import numpy as np

from view_generation.geometric_parameters import convex_hull_vs_grid, ratio_bounding_convex_hull, \
    fibonacci_unit_sphere, gaussian_histogram, shell_histogram, sector_histogram
from view_generation.pca_based import pca_attributes
from view_generation.pcl_descriptors import esf_descriptor, pfh_global_radius


class Views:
    def __init__(self, data_object):
        self.dataset_name = data_object.dataset_name  # dataset_name
        self.number_of_sampled_points = data_object.number_of_sampled_points
        self.view_path = data_object.view_path

        self.data_object = data_object

        # histograms
        self.esfs = None
        self.global_pfhs = None
        self.gauss = None
        self.shells = None
        self.sectors = None

        # 3-dim
        self.pca_explained_variance_ratios = None

        # 1-dim
        self.n_segments_for_samp = data_object.n_segments_for_samp
        self.samp = []

        self.n_grids_for_grid_vs_convex_hull = data_object.n_grids_for_grid_vs_convex_hull
        self.convex_hull_vs_grid = []

        self.ratio_bounding_volume_convex_hull = None

        self.prefix = self.view_path + self.dataset_name + '_' + str(self.number_of_sampled_points) \
                      + '_'

    def read_in_views(self):

        try:
            self.esfs = np.load(self.prefix + 'esf_histograms.npy')
        except FileNotFoundError:
            print('File for ESF histograms not found.')

        try:
            self.global_pfhs = np.load(self.prefix + 'pfh_histograms.npy')
        except FileNotFoundError:
            print('File for PFH histograms not found.')

        try:
            self.pca_explained_variance_ratios = np.load(
                self.prefix + 'pca_explained_variance_ratio.npy')
        except FileNotFoundError:
            print('File for PCA explained variance ratio not found.')

        try:
            self.ratio_bounding_volume_convex_hull = np.load(
                self.prefix + 'ratio_bounding_volume_convex_hull.npy')
        except FileNotFoundError:
            print('File for Ratio Bounding Volume vs Convex Hull not found.')

        try:
            self.sectors = np.load(
                self.prefix + 'sector_histograms_' + str(self.data_object.n_sectors) + '.npy')
        except FileNotFoundError:
            print('File for Sector Histogram not found.')

        self.samp = []
        for n in self.n_segments_for_samp:
            file = self.prefix + 'samp_min_max_normed_scaled_' + str(n) + '.npy'
            try:
                self.samp.append(np.load(file))
            except FileNotFoundError:
                print('No SAMP file found:', file)

        self.convex_hull_vs_grid = []
        for n in self.data_object.n_grids_for_grid_vs_convex_hull:
            file = self.prefix + 'convex_hull_vs_grid_' + str(n) + '.npy'
            try:
                self.convex_hull_vs_grid.append(np.load(file))
            except FileNotFoundError:
                print('No Convex Hull vs. Grid file found:', file)

        self.gauss = []
        for n in self.data_object.n_points_on_sphere_for_gauss:
            file = self.prefix + 'gauss_histogram_' + str(n) + '.npy'
            try:
                self.gauss.append(np.load(file))
            except FileNotFoundError:
                print('No Gauss histogram file found:', file)

        self.shells = []
        for n in self.data_object.n_shells:
            file = self.prefix + 'shell_histogram_' + str(n) + '.npy'
            try:
                self.shells.append(np.load(file))
            except FileNotFoundError:
                print('No Shell histogram file found:', file)

    def generate_and_save_esf(self):
        esf_histograms = [esf_descriptor(pc) for pc in self.data_object.pcl_pointclouds]
        np.save(self.prefix + 'esf_histograms', esf_histograms)

    def generate_and_save_global_pfh_with_external_normals(self):
        if not self.data_object.normals_as_arrays:
            print('Cannot compute pfh without normals.')
            return
        print('Compute PFH histograms...')
        pfh_histograms = [pfh_global_radius(pc, prec_normals=np.asarray(
            self.data_object.o3d_pointclouds[i].normals)) for i, pc in tqdm(enumerate(
            self.data_object.pcl_pointclouds))]
        np.save(self.prefix + 'pfh_histograms', pfh_histograms)

    def generate_and_save_pca_explained_variance(self):
        pca_singular_values, pca_explained_variance, pca_explained_variance_ratio = pca_attributes(
            [np.asarray(pcd.points) for pcd in self.data_object.o3d_pointclouds])
        np.save(self.prefix + 'pca_explained_variance_ratio', pca_explained_variance_ratio)

    def generate_and_save_grid_vs_convex_hull(self):
        for n in self.data_object.n_grids_for_grid_vs_convex_hull:
            convex_hull_vs_grid_res, _ = convex_hull_vs_grid(
                self.data_object.varimax_projections_2d, number_of_bins=n, draw=2)
            np.save(self.prefix + 'convex_hull_vs_grid_' + str(n), convex_hull_vs_grid_res)

    def generate_and_save_ratio_bounding_volume_convex_hull(self):
        ratio_bounding_area_convex_hull_res, ratio_bounding_volume_convex_hull_res = \
            ratio_bounding_convex_hull(
            self.data_object.varimax_projections_2d, draw=True)
        np.save(self.prefix + 'ratio_bounding_volume_convex_hull',
                ratio_bounding_volume_convex_hull_res)

    def generate_and_save_gauss_histogram(self):
        print('Generate and save Gauss histograms...')
        for n in self.data_object.n_points_on_sphere_for_gauss:
            sphere = np.array(fibonacci_unit_sphere(n))
            gauss_histograms = [gaussian_histogram(
                pc.points, self.data_object.normals_as_arrays[i], sphere) for i, pc in tqdm(
                enumerate(self.data_object.o3d_pointclouds))]
            np.save(self.prefix + 'gauss_histogram_' + str(n), gauss_histograms)

    def generate_and_save_shell_histogram(self):
        for n in self.data_object.n_shells:
            shell_histograms = [shell_histogram(pc.points, n) for i, pc in
                                tqdm(enumerate(self.data_object.o3d_pointclouds))]
            np.save(self.prefix + 'shell_histogram_' + str(n), shell_histograms)

    def generate_and_save_sector(self):
        print('Generate and save sector histograms...')
        sector_histograms = [sector_histogram(np.asarray(pc.points), self.data_object.n_sectors)
                             for pc in tqdm(self.data_object.o3d_pointclouds)]
        np.save(self.prefix + 'sector_histograms_' + str(self.data_object.n_sectors),
                sector_histograms)

class Mvsc3dData:
    def __init__(self,
                 dataset_name,
                 number_of_sampled_points,
                 generated_path='../generated/',
                 n_segements_for_samp=None,
                 n_grids_for_grid_vs_convex_hull=None,
                 n_points_on_sphere_for_gauss=None,
                 n_shells=None,
                 n_sectors=8):
        if n_grids_for_grid_vs_convex_hull is None:
            n_grids_for_grid_vs_convex_hull = [30, 60]
        if n_segements_for_samp is None:
            n_segements_for_samp = [20]
        if n_points_on_sphere_for_gauss is None:
            n_points_on_sphere_for_gauss = [33, 66]
        if n_shells is None:
            n_shells=[5, 10, 20]
        self.dataset_name = dataset_name
        self.number_of_sampled_points = number_of_sampled_points
        self.labels = None
        self.meshes = None
        self.o3d_pointclouds = None
        self.pcl_pointclouds = None
        self.varimax_projections_2d = None
        self.index_changes = None
        self.normals_as_arrays = None
        self.generated_path = generated_path
        self.view_path = self.generated_path + dataset_name + '/views/' + 'poisson_' + \
                         str(number_of_sampled_points) + '/'

        # for views
        self.n_segments_for_samp = n_segements_for_samp
        self.n_grids_for_grid_vs_convex_hull = n_grids_for_grid_vs_convex_hull
        self.n_points_on_sphere_for_gauss = n_points_on_sphere_for_gauss
        self.n_shells = n_shells
        self.n_sectors = n_sectors
        self.views = Views(self)

    # def read_in_data(self):
    #     DATA_SET_NAME = self.dataset_name
    #     meshes, labels = o3d_meshes(DATA_SET_NAME)  # , '../../data')
    #     self.meshes = meshes
    #     self.labels = labels
    # 
    #     pcl_point_clouds, o3d_pointclouds = read_pcd_pointclouds_poisson(
    #         DATA_SET_NAME, self.number_of_sampled_points,
    #         path_to_generated_folder=self.generated_path)
    #     self.pcl_pointclouds = pcl_point_clouds
    #     self.o3d_pointclouds = o3d_pointclouds
    # 
    #     index_changes = np.where(np.array(labels)[:-1] != np.array(labels)[1:])[0].tolist()
    #     index_changes = [0] + index_changes + [len(labels) - 1]
    #     self.index_changes = index_changes
    # 
    #     if self.pointclouds_have_normals():
    #         print('Normals present in Open3D pointclouds.')
    #         self.normals_as_arrays = [np.asarray(pc.normals) for pc in o3d_pointclouds]
    # 
    #     # read in Varimax projections 2D
    #     varimax_file = self.generated_path + self.dataset_name +'/' + self.dataset_name + '_varimax_2d_poisson_' + \
    #                    str(self.number_of_sampled_points) + '.npy'
    #     #../generated/mc_gill/mc_gill_varimax_2d_poisson_5000.npy
    #     try:
    #         self.varimax_projections_2d = np.load(varimax_file)
    #     except FileNotFoundError:
    #         print('Could not find file ' + varimax_file)
    #         print('Could not read in 2D Varimax projections. Please use '
    #               'https://github.com/lkrmbhlz/samp to get Varimax projections.')

    def pointclouds_have_normals(self):
        try:
            normals = np.asarray(self.o3d_pointclouds[0].normals)
            if len(normals) == 0:
                print('No normals present in Open3D pointclouds. To estimate normals from the '
                      'points, call estimate_normals_for_03d_pointclouds.')
                return False
            else:
                return True
        except IndexError:
            return False
        except TypeError:
            print('No Open 3D pointcloud objects present. Call read_in_data().')
            return False

    def estimate_normals_for_o3d_pointclouds(self):
        for o3d_pc in tqdm(self.o3d_pointclouds):
            o3d_pc.estimate_normals()
        self.normals_as_arrays = [np.asarray(pc.normals) for pc in self.o3d_pointclouds]

    def generate_and_save_views(self):
        pass
        #self.views.generate_and_save_esf()
        #self.views.generate_and_save_global_pfh_with_external_normals()
        #self.views.generate_and_save_pca_explained_variance()
        #self.views.generate_and_save_grid_vs_convex_hull()
        #self.views.generate_and_save_ratio_bounding_volume_convex_hull()
        #self.views.generate_and_save_gauss_histogram()
        #self.views.generate_and_save_shell_histogram()
        #self.views.generate_and_save_sector()


    def get_generated_views(self):
        self.views.read_in_views()
        #print('Views Read')
        views = {'ESF': self.views.esfs, 'Global PFH': np.squeeze(self.views.global_pfhs),
                 'Explained Variance Ratio of PCA': self.views.pca_explained_variance_ratios,
                 'Ratio Bounding Volume Convex Hull': self.views.ratio_bounding_volume_convex_hull,
                 'Sector Histograms': self.views.sectors,
                 'SAMP': {},
                 'Grid vs. Convex Hull': {},
                 'Gauss Maps': {},
                 'Shell Histograms': {}}

        for i, n in enumerate(self.n_segments_for_samp):
            views['SAMP']['N Segments ' + str(n)] = self.views.samp[i]
        for i, n in enumerate(self.n_grids_for_grid_vs_convex_hull):
            views['Grid vs. Convex Hull']['N Grids ' + str(n)] = self.views.convex_hull_vs_grid[i]
        try:
            for i, n in enumerate(self.n_points_on_sphere_for_gauss):
                views['Gauss Maps']['N Points On Sphere ' + str(n)] = self.views.gauss[i]
        except IndexError:
            print('No Gauss Views')
        try:
            for i, n in enumerate(self.n_shells):
                views['Shell Histograms']['N Shells ' + str(n)] = self.views.shells[i]
        except IndexError:
            print('No Shell Histograms View')

        return views


#data = Mvsc3dData('mc_gill', 5000)
#data.read_in_data()
#data.estimate_normals_for_o3d_pointclouds()
#data.generate_and_save_views()
# data.estimate_normals_for_o3d_pointclouds()
# print(data.pointclouds_have_normals())
