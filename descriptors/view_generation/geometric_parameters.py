import math

import open3d as o3d
from scipy.spatial import ConvexHull, distance
import numpy as np
from sklearn.neighbors._kd_tree import KDTree
from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.spatial.distance import euclidean
from sklearn.decomposition import PCA
import numpy as np


def bins_stat(x, y, number_of_bins):
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    binx = np.linspace(xmin, xmax, number_of_bins)
    biny = np.linspace(ymin, ymax, number_of_bins)
    # size = euclidean(binx[0],binx[1])**2
    size = euclidean(binx[0], binx[1]) * euclidean(biny[0], biny[1])
    # print('x: ',euclidean(binx[0],binx[1]))
    # print('y: ',euclidean(biny[0], biny[1]))

    return size, stats.binned_statistic_2d(x, y, None, 'count', bins=[binx, biny]), binx, biny


def ratio_bounding_convex_hull(projected_points_list, draw=False):
    convex_hulls_varimax = []
    areas_varimax = []
    volumes_varimax = []
    for i, points in enumerate(projected_points_list):
        # if all projected point lie in one dimension, add some randomness so convex hull can be
        # calculated otherwise QhullError: QH6154 Qhull precision error: will be thrown
        if np.all(points[:, 1] == 0):
            old_value_x, old_value_y = points[0]
            points[0] = [old_value_x, old_value_y + 0.0000000000001]
            # print('------------->', points[0])
        hull = ConvexHull(points)
        convex_hulls_varimax.append(hull)
        areas_varimax.append(hull.area)
        volumes_varimax.append(hull.volume)

    ratio_bounding_area_convex_hull = []
    ratio_bounding_volume_convex_hull = []

    if draw:
        fig, axs = plt.subplots(2,2, figsize=(10,10))
    for i, points in enumerate(projected_points_list):
        points = np.asarray(points)

        x_coordinates = np.asarray(points[:, 0])
        y_coordinates = np.asarray(points[:, 1])

        area_bounding_box = abs(max(x_coordinates) - min(x_coordinates)) * abs(
                    max(y_coordinates) - min(y_coordinates))
        perimeter_bounding_box = (abs(max(x_coordinates) - min(x_coordinates)) + abs(
                max(y_coordinates) - min(y_coordinates))) * 2

        ratio_area = areas_varimax[i] / perimeter_bounding_box
        ratio_volume = volumes_varimax[i] / area_bounding_box

        ratio_bounding_area_convex_hull.append(ratio_area)
        ratio_bounding_volume_convex_hull.append(ratio_volume)

        if draw and i<=1:
            axs[i][0].scatter(points[:,0], points[:,1], marker='.', c='black', alpha=0.002)
            for simplex in convex_hulls_varimax[i].simplices:
                # print(simplex)
                axs[i][0].plot(points[simplex, 0], points[simplex, 1], '--', c='r')
            bbox= np.array([[max(x_coordinates), min(y_coordinates)],
                   [max(x_coordinates), max(y_coordinates)],
                   [min(x_coordinates), max(y_coordinates)],
                   [min(x_coordinates), min(y_coordinates)],
                   [max(x_coordinates), min(y_coordinates)]])
            axs[i][0].plot(bbox[:,0], bbox[:,1])
            #for j, p in enumerate([0,1,2,3]):
            #    axs[i][0].plot(bbox[j], bbox[j+1])
            axs[i][0].scatter(max(x_coordinates), min(y_coordinates), c='b')
            axs[i][0].scatter(max(x_coordinates), max(y_coordinates), c='b')
            axs[i][0].scatter(min(x_coordinates), max(y_coordinates), c='b')
            axs[i][0].scatter(min(x_coordinates), min(y_coordinates), c='b')

    return ratio_bounding_area_convex_hull, ratio_bounding_volume_convex_hull


def convex_hull_parameters(points_list):
    convex_hulls = []
    areas = []
    volumes = []
    for points in points_list:
        # if all projected point lie in one dimension, add some randomness so convex hull can be
        # calculated otherwise QhullError: QH6154 Qhull precision error: will be thrown
        if np.all(points[:, 1] == 0):
            old_value_x, old_value_y = points[0]
            points[0] = [old_value_x, old_value_y + 0.0000000000001]
        hull = ConvexHull(points)
        convex_hulls.append(hull)
        areas.append(hull.area)
        volumes.append(hull.volume)
    #print(convex_hulls[0].vertices)
    #print(convex_hulls[3].vertices)
    return areas, volumes, convex_hulls


def ratio_convex_hull_calculated_mesh(points_list, alpha):
    ratios = []
    for object in tqdm(points_list):
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(points_list, alpha)

        convex_hull, _ = mesh.compute_convex_hull()
        convex_hull_volume = convex_hull.get_volume()

        if mesh.is_watertight():
            mesh_volume = mesh.get_volume()
        else:
            print('not_watertight')
            mesh_volume = convex_hull_volume
        ratio = mesh_volume / convex_hull_volume
        ratios.append(ratio)
    return ratios


def convex_hull_vs_grid(projections, number_of_bins=30, draw=5, drawe=True):
    areas, volumes, convex_hulls = convex_hull_parameters(projections)

    no_points_in_grid = []
    convex_hull_vs_grid = []
    #fig, axs = plt.subplots(draw, 2, figsize=(100, 50*draw))

    for i, projection in enumerate(projections):
        size, bins, binx, biny = bins_stat(projection[:, 0], projection[:, 1],
                                           number_of_bins=number_of_bins)
        bins = bins.statistic
        bins = [item for sublist in bins for item in sublist]
        no_points = len([x for x in bins if x == 0])
        points = len([x for x in bins if x != 0])

        area_with_points = points * size

        #convex_hull_vs_grid.append(abs(volumes[i] - area_with_points))
        convex_hull_vs_grid.append(area_with_points / volumes[i])
        no_points_in_grid.append(no_points)

        #if i < draw and drawe:
        #    axs[i][0].scatter(projection[:, 0], projection[:, 1], marker='.', alpha=1, c='black',
        #                      s=0.5)
        #    axs[i][0].set_aspect('equal')
        #    axs[i][0].hlines(biny, min(binx), max(binx), colors=(0, 0, 0, 0.1))
        #    axs[i][0].vlines(binx, min(biny), max(biny), colors=(0, 0, 0, 0.1))
        #    hull = convex_hulls[i]
        #    #axs[i][0].plot(projection[hull.vertices, 0], projection[hull.vertices, 1], 'r--')
        #    for simplex in hull.simplices:
        #        #print(simplex)
        #        axs[i][0].plot(projection[simplex, 0], projection[simplex, 1], '--', c='r')
#
        #    axs[i][0].set_xlabel('x', size=15)
        #    axs[i][0].set_ylabel('y', size=15)
        #    #axs[i][1].set_xlabel('convex hull volume')
        #    axs[i][1].set_ylabel('Area', size=15)
        #    # axs[i][1].set_xlim(xmin=0, xmax=10000)
        #    # axs[i][1].set_ylim(ymin=0, ymax=10000)
#
        #    # axs[i][1].scatter(abs(volumes_pca[i]), area_with_points)
        #    axs[i][1].bar([1, 2], [volumes[i], area_with_points])
        #                  #tick_label=['Convex Hull', 'Bins with Points'])
        #    axs[i][1].set_xticks([1,2])
        #    axs[i][1].set_xticklabels(['Convex Hull', 'Bins with Points'], size=15)
    return convex_hull_vs_grid, no_points_in_grid


def gaussian_histogram(points, normals, uniformely_sampled_points_on_sphere):
    _, aligned_normals = align_point_normals_with_principal_components(points, normals)

    # alternative to NN-query: Dot product between rotated normals and points on sphere?
    # -> Smallest angle. Faster?
    kdtree = KDTree(uniformely_sampled_points_on_sphere)
    histogram = np.zeros(len(uniformely_sampled_points_on_sphere))
    for normal in aligned_normals:
        x, y, z = normal
        d, i = kdtree.query([[x, y, z]])
        #print(i)
        histogram[i] = histogram[i] + 1
    return histogram / len(aligned_normals)


def shell_histogram(points, n_shells):
    lower = np.min(points, axis=0)  # lower left corner of bounding box
    upper = np.max(points, axis=0)  # upper right corner of bounding box

    centroid = np.array(points).mean(axis=0)
    #print(centroid)
    # c = pcl.vectors.Float([centroid[0], centroid[1], centroid[2]])
    centered_points = points - centroid

    max_r = distance.euclidean(lower, upper) / 2
    shell_segments = np.linspace(0,max_r, n_shells)
    #print(shell_segments)
    histogram = np.zeros(len(shell_segments)-1)

    for i,_ in enumerate(shell_segments[:-1]):
        min = shell_segments[i]
        max = shell_segments[i+1]
        for point in centered_points:
            if min <= distance.euclidean([0,0,0], point) < max:
                histogram[i] = histogram[i] + 1
    histogram = histogram / len(points)
    return histogram


def sector_histogram(points, n_sectors):
    #print(np.array(points))

    # align with components
    pca = PCA(n_components=3).fit(points)
    rotated_points = pca.transform(points)
    points = rotated_points
    #print(points)

    centroid = np.array(points).mean(axis=0)
    centered_points = points - centroid
    #print(centered_points)
    sectors = np.array(fibonacci_unit_sphere(n_sectors))
    #print(sectors)
    # dot product of each point with fib, take min
    distances = np.dot(rotated_points, sectors.T)
    closest_per_point = np.argmax(distances, 1)
    #print('----closest---')
    #print(closest_per_point)

    #print('---bincount---')
    #print(np.bincount(closest_per_point))

    histogram = np.bincount(closest_per_point, minlength=len(sectors)) / len(points)
    #print(histogram)
    return histogram
    #print(distances)

    #print('----hist---')
    #print(histogram)
    #sorted = np.sort(histogram)[::-1]
    ##print('----hist sort')
    ##print(sorted)
    #return sorted


#def extended_gaussian_histogram(Points, Normals, uniformely_sampled_points_on_sphere):
#    S = uniformely_sampled_points_on_sphere.shape[1]
#    hist = np.zeros(S)
#    _, rotated = align_point_normals_with_principal_components(Points, Normals)
#    D = np.dot(np.array(rotated), np.array(uniformely_sampled_points_on_sphere).T) # N x M
#    print(D)
#    nearest = np.argmax(D, 1) # for each normal, the index of nearest spherical direction
#    count = np.bincount(nearest) # number of normals associated with each direction
#    print(count)
#    hist[:count.shape[0]] = count
#    return hist

def align_point_normals_with_principal_components(points, normals):
    #A = np.dot(points, points.T)
    #[eigs, eigenvectors] = np.linalg.eig(A)
    pca = PCA().fit(points)
    rotated_points = pca.transform(points)
    rotated_normals_2 = np.dot(normals, pca.components_)
    #rotated_normals = pca.transform(normals)
    #rotated_normals = np.dot(eigenvectors.T, normals)
    #rotated_points = np.dot(eigenvectors.T, points)
    return rotated_points, rotated_normals_2


def fibonacci_unit_sphere(samples=1000):

    points = []
    phi = math.pi * (3. - math.sqrt(5.))  # golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # h_k y goes from 1 to -1
        radius = math.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = math.cos(theta) * radius
        z = math.sin(theta) * radius

        points.append([x, y, z])

    return np.array(points)

def fibonacci_unit_sphere_variation(samples=50):
    n = samples
    goldenRatio = (1 + 5 ** 0.5) / 2
    i = np.arange(0, n)
    theta = 2 * np.pi * i / goldenRatio
    phi = np.arccos(1 - 2 * (i + 0.5) / n)
    x, y, z = np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi);
    return np.array(list(zip(x, y, z)))

#varimax_projections_2d = np.load(
#    '../generated/modelnet10/modelnet10_varimax_1st_2nd_poisson_10000.npy')

#ratio_bounding_area_convex_hull_res, ratio_bounding_volume_convex_hull_res = ratio_bounding_convex_hull(
#    varimax_projections_2d)
#p = sphere_points = fibonacci_unit_sphere(4)
#h = shell_histogram(p, 5)
#print(h)

#pc = extended_gaussian_histogram(np.array([[0,1,1], [0,2,2]]), np.array([[0.2,0.3,0.5],[0.5,0.3,0.2]]), np.array([[0,0,0]]))
#pc = [[1,1,1], [3,3,3], [10,10,10]] # [0,1,1], [0,2,2],
#sector_histogram(pc, 3)

#print(fibonacci_unit_sphere(5))
#print('---')
#print(fibonacci_unit_sphere_variation(5))