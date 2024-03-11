"""Point cloud processing utilities."""

import numpy as np
import open3d as o3d
import sklearn


def load_ply(path):
    """Load point cloud from ply file."""
    pcd = o3d.io.read_point_cloud(path)
    return pcd


def to_array(tensor):
    """Conver tensor to array."""
    if (not isinstance(tensor, np.ndarray)):
        return np.as_array(tensor)
    else:
        return tensor


def to_o3d_vec(vec):
    """Convert to open3d vector objects."""
    return o3d.utility.Vector3dVector(to_array(vec))


def to_o3d_pcd(xyz):
    """Convert array to open3d PointCloud.

    Args
    ----
        xyz (np.ndarray): The input point cloud array in [N, 3].
    
    Returns
    -------
        pcd (open3d.geometry.PointCloud): Open3d point cloud object.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(
        to_array(xyz).astype(np.float64)
    )
    return pcd


def to_o3d_feats(embedding):
    """Convert embedding array to open3d features.

    Args
    ----
        embedding (np.ndarray): Embedding array of [N, D].

    Returns
    -------
        feats (open3d.registration.Feature): Open3d feature object.
    """
    feats = o3d.registration.Feature()
    feats.data = to_array(embedding).T
    return feats


def transform_points(points, trans):
    """Transform points using the given transformation matrix.

    Args
    ----
        points (np.ndarray): The input points in [N, 3].
        trans (np.ndarray): The transformation matrix in [4, 4].

    Returns
    -------
        np.ndarray: The transformed points in [N, 3].
    """
    return np.dot(points, trans[:3, :3].T) + trans[:3, 3]


class PointCloudCache:
    """Cache for point clouds to avoid redundant loading. Singleton pattern."""

    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(PointCloudCache, cls).__new__(cls)
            cls.instance.point_cloud_cache = {}
            cls.instance.kd_tree_cache = {}
        return cls.instance
    
    def load(self, path):
        """Load a point cloud from the given path."""
        if path not in self.point_cloud_cache:
            pcd = load_ply(path)
            self.point_cloud_cache[path] = np.asarray(pcd.points)
        return self.point_cloud_cache[path]
    
    def get_tree(self, path):
        """Get the KD tree for the point cloud at the given path."""
        if path not in self.kd_tree_cache:
            points = self.load(path)
            self.kd_tree_cache[path] = sklearn.neighbors.KDTree(points)
        return self.kd_tree_cache[path]
    
    def clear(self):
        """Clear the cache."""
        self.point_cloud_cache.clear()
        self.kd_tree_cache.clear()


def get_correspondences(src_path, tgt_path, trans, dist_thresh=0.1):
    """Get correspondences point pairs between two point clouds.

    Args
    ----
        src_path (str): Path to the source point cloud file.
        tgt_path (str): Path to the target point cloud file.
        trans (np.ndarray): The 4x4 transformation matrix from source to target.
        dist_thresh (float): The distance threshold for overlapping points. Defaults to 0.1.

    Returns
    -------
        np.ndarray: The indices of correspondences in the source and target point clouds.
    """
    point_cloud_cache = PointCloudCache()
    src_points = point_cloud_cache.load(src_path)
    src_points_transformed = transform_points(src_points, trans)

    tree = point_cloud_cache.get_tree(tgt_path)
    dist, indices = tree.query(src_points_transformed, k=1, return_distance=True)
    correspondences = np.hstack((np.arange(len(src_points))[:, np.newaxis], indices))
    correspondences = correspondences[dist[:, 0] < dist_thresh]
    return correspondences
