from tqdm import tqdm
import numpy as np

from descriptors.view_generation.geometric_parameters import convex_hull_vs_grid, ratio_bounding_convex_hull
from descriptors.view_generation.projection_based import varimax_projections_2d, samp
from descriptors.view_generation.pca_based import pca_attributes


class Views:
    def __init__(self, data_object):
        self.dataset_name = data_object['dataset_name']  # dataset_name
        # self.number_of_sampled_points = data_object.number_of_sampled_points

        self.data = data_object['point_clouds']
        self.labels = data_object['labels']

        self.evrap = None
        self.samp = None
        self.scomp = None
        self.sirm = None

        self.prefix = self.dataset_name + '/'

    def read_in_views(self):
        try:
            self.evrap = np.load(self.prefix + 'evrap.npy')
        except FileNotFoundError:
            print('File for EVRAP.')

        try:
            self.sirm = np.load(self.prefix + 'sirm.npy')
        except FileNotFoundError:
            print('File for SIRM.')

        try:
            self.samp = np.load(self.prefix + 'samp.npy')
        except FileNotFoundError:
            print('File for SAMP not found.')

        try:
            self.scomp = np.load(self.prefix + 'scomp.npy')
        except FileNotFoundError:
            print('File for SCOMP not found.')

    def generate_all_descriptors(self):
        self.generate_and_save_evrap()
        self.generate_and_save_scomp()
        self.generate_and_save_sirm()
        self.generate_and_save_samp()

    def generate_and_save_evrap(self):
        _, _, explained_var_ratio = pca_attributes([np.asarray(self.data[i,:]) for i in range(len(self.data))])
        np.save(self.prefix + 'evrap.npy', explained_var_ratio)

    def generate_and_save_scomp(self):
        varimax_projection = varimax_projections_2d([np.asarray(self.data[i, :]) for i in range(len(self.data))])
        convex_hull_vs_grid_res, _ = convex_hull_vs_grid(varimax_projection, number_of_bins=20, draw=2)
        np.save(self.prefix + 'scomp.npy', convex_hull_vs_grid_res)

    def generate_and_save_sirm(self):
        _, sirm_data = ratio_bounding_convex_hull(
            varimax_projections_2d(self.data), draw=False
        )
        np.save(self.prefix + 'sirm.npy', sirm_data)

    def generate_and_save_samp(self):
        samp_data = samp(self.data)
        np.save(self.prefix, 'data.npy', samp_data)


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
                 'Explained Variance Ratio of PCA': self.views.evrap,
                 'Ratio Bounding Volume Convex Hull': self.views.sirm,
                 'Sector Histograms': self.views.sectors,
                 'SAMP': {},
                 'Grid vs. Convex Hull': {},
                 'Gauss Maps': {},
                 'Shell Histograms': {}}

        for i, n in enumerate(self.n_segments_for_samp):
            views['SAMP']['N Segments ' + str(n)] = self.views.samp[i]
        for i, n in enumerate(self.n_grids_for_grid_vs_convex_hull):
            views['Grid vs. Convex Hull']['N Grids ' + str(n)] = self.views.scomp[i]
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
