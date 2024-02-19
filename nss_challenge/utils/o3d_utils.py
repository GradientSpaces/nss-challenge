import numpy as np
import open3d as o3d
import copy


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

    Args:
        xyz (np.ndarray): The input point cloud array in [N, 3].
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(
        to_array(xyz).astype(np.float64)
    )
    return pcd


def to_o3d_feats(embedding):
    """Convert embedding array to open3d features.

    Args:
        embedding (np.ndarray): Embedding array of [N, D].
    """
    feats = o3d.registration.Feature()
    feats.data = to_array(embedding).T
    return feats


def get_correspondences(src_pcd, tgt_pcd, trans, search_voxel_size, k=None):
    """Get correspondences."""

    # use deepcopy to avoid overwrite raw data
    src_pcd_tmp = copy.deepcopy(src_pcd)
    tgt_pcd_tmp = copy.deepcopy(tgt_pcd)
    src_pcd_tmp.transform(trans)
    pcd_tree = o3d.geometry.KDTreeFlann(tgt_pcd_tmp)
    correspondences = []
    for i, point in enumerate(src_pcd_tmp.points):
        _, idx, _ = pcd_tree.search_radius_vector_3d(point, search_voxel_size)
        if k is not None:
            idx = idx[:k]
        for j in idx:
            correspondences.append([i, j])
    return  np.array(correspondences)
