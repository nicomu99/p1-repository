import pclpy
from pclpy import pcl
import numpy as np
from scipy.spatial import distance

def global_fpfh(pointcloud):
    gfpfh = pcl.features.GFPFHEstimation.PointXYZ_PointXYZL_GFPFHSignature16()

    pass


def fpfh_global_radius(point_cloud, normals_radius, prec_normals=None):
    lowee = np.min(point_cloud.xyz, axis=0)  # lower left corner of bounding box
    upper = np.max(point_cloud.xyz, axis=0)  # upper right corner of bounding box
    max_r = distance.euclidean(lowee, upper)

    if prec_normals is None:
        ne = pcl.features.NormalEstimation.PointXYZ_Normal()
        ne.setInputCloud(point_cloud)

        tree = pcl.search.KdTree.PointXYZ()
        ne.setSearchMethod(tree)

        normals = pcl.PointCloud.Normal()
        ne.setRadiusSearch(normals_radius)
        # Calculate normals
        ne.compute(normals)
    else:
        normals = pcl.PointCloud.Normal().from_array(prec_normals)

    # Construct FPFH estimation class and pass cloud and normals in
    fpfh = pcl.features.FPFHEstimation.PointXYZ_Normal_FPFHSignature33()
    fpfh.setInputCloud(point_cloud)
    fpfh.setInputNormals(normals)
    # Or, if the cloud is of PointNormal type, execute fpfh.setInputNormals(cloud);

    # Construct a kd tree
    # Its contents will be filled into the object according to the given input point cloud (because no other search surface is given).
    tree = pcl.search.KdTree.PointXYZ()
    fpfh.setSearchMethod(tree)

    # output
    pfhs = pcl.PointCloud.FPFHSignature33()

    # Use neighbor points within a 5cmm sphere
    # Note: the radius used here must be greater than the radius used to estimate the surface normal!!
    fpfh.setRadiusSearch(max_r)

    #indices = pcl.vectors.Int([1,500, 1000, 40000])
    indices = pcl.vectors.Int([np.random.randint(0, normals.size()-1)])
    fpfh.setIndices(indices)

    # Calculation characteristics
    fpfh.compute(pfhs)
    return pfhs.histogram


def pfh_global_radius(point_cloud, normal_radius=None, prec_normals=None):
    lowee = np.min(point_cloud.xyz, axis=0)  # lower left corner of bounding box
    upper = np.max(point_cloud.xyz, axis=0)  # upper right corner of bounding box
    max_r = distance.euclidean(lowee, upper)
    if prec_normals is None:
        ne = pcl.features.NormalEstimation.PointXYZ_Normal()
        ne.setInputCloud(point_cloud)

        tree = pcl.search.KdTree.PointXYZ()
        ne.setSearchMethod(tree)

        normals = pcl.PointCloud.Normal()
        ne.setRadiusSearch(normal_radius)
        ne.compute(normals)
    else:
        normals = pcl.PointCloud.Normal().from_array(prec_normals)
        #vfh.setInputNormals(norm)

    pfh = pcl.features.PFHEstimation.PointXYZ_Normal_PFHSignature125()
    pfh.setInputCloud(point_cloud)
    pfh.setInputNormals(normals)
    indices = pcl.vectors.Int([np.random.randint(0, normals.size()-1)])
    pfh.setIndices(indices)

    tree = pcl.search.KdTree.PointXYZ()
    pfh.setSearchMethod(tree)
    pfhs = pcl.PointCloud.PFHSignature125()
    pfh.setRadiusSearch(max_r)
    pfh.compute(pfhs)
    #f1 = 0.0
    #f2 = 0.0
    #f3 = 0.0
    #f4 = 0.0
    #pfh.computePairFeatures(point_cloud, normals, indices[0], 1,  f1, f2, f3, f4)

    return pfhs.histogram


def vfh_descriptor(point_cloud, radius, set_viewpoint_to_centroid=False, center=False, normals=None,
                   translate=None):

    if center:
        centroid = point_cloud.xyz.mean(axis=0)
        print(centroid)
        #c = pcl.vectors.Float([centroid[0], centroid[1], centroid[2]])
        centered_points = point_cloud.xyz - centroid
        point_cloud_centered = pcl.PointCloud.PointXYZ.from_array(centered_points)
        print(point_cloud_centered.xyz.mean(axis=0))

    ne = pcl.features.NormalEstimation.PointXYZ_Normal()
    if center:
        ne.setInputCloud(point_cloud_centered)
    else:
        ne.setInputCloud(point_cloud)
    tree = pcl.search.KdTree.PointXYZ()
    ne.setSearchMethod(tree)
    cloud_normals = pcl.PointCloud.Normal()
    ne.setRadiusSearch(radius)  # 0.9

    ne.compute(cloud_normals)

    tree = pcl.search.KdTree.PointXYZ()

    # estimate vfh
    vfh = pcl.features.VFHEstimation.PointXYZ_Normal_VFHSignature308()
    if translate is not None:
        points_moved = point_cloud_centered.xyz + [500, 500, 500]
        point_cloud_moved = pcl.PointCloud.PointXYZ.from_array(points_moved)
        vfh.setInputCloud(point_cloud_moved)
        print(point_cloud_moved.xyz.mean(axis=0))
        print('--')
    else:
        vfh.setInputCloud(point_cloud)

    if normals is not None:
        norm = pcl.PointCloud.Normal().from_array(normals)
        vfh.setInputNormals(norm)
    else:
        vfh.setInputNormals(cloud_normals)
    if set_viewpoint_to_centroid:
        centroid = point_cloud.xyz.mean(axis=0)
        c = pcl.vectors.Float([centroid[0],centroid[1],centroid[2]])
        vfh.setViewPoint(c[0], c[1], c[2])
    vfh.setSearchMethod(tree)
    output = pcl.PointCloud.VFHSignature308()
    vfh.compute(output)

    return output.histogram[0]


def esf_descriptor(point_cloud):
    """
    Notes
    "The ESF consists of ten 64-bin-sized histograms resulting in a single 640 value histogram for
    input point cloud. It is a combination of three shape functions [15] point distance (D2), angle (A3), and
    area (D3). ESF global descriptor is unique since it does not require the use of normals information to
    describe the cloud.
    In this descriptor, a voxel grid algorithm is used as an approximation of the real surface. A voxel
    grid iterates over each the points in the cloud, for each iteration, three random points are chosen. For
    these points, the shape functions are computed:
     D2 computes the distances between two randomly selected points. After that, for each pair of
    points, it checks if the connecting lines between points reside: on the surface, off the surface,
    or mixed. Based on this, the distance value will be binned to one of the three possible
    histograms: on, off or mixed.
     A3 computes the angle enclosed by the points. Next, the value is binned based on how the line
    opposite to the angle is (on, off or mixed).
     D3 computes the area of the triangle formed by the 3 points. Similar to D2, the result is also
    classified as on, off or mixed."[#1]_

    References
    ----------
    .. [#1] Alhamzi, Khaled & Elmogy, Mohammed & Barakat, Sherif. (2015). 3D Object Recognition Based on Local and Global Features Using Point Cloud Library. International Journal of Advancements in Computing Technology. 7. 43-54.

    Parameters
    ----------
    point_cloud

    Returns
    -------
    a 640 value histogram

    """
    esf = pcl.features.ESFEstimation.PointXYZ_ESFSignature640()
    esf.setInputCloud(point_cloud)

    output = pcl.PointCloud.ESFSignature640()
    esf.compute(output)

    return output.histogram[0]
