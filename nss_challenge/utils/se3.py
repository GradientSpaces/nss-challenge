"""SE3 utility functions."""

import random

import numpy as np


def uniform_2_sphere(num: int = None):
    """Uniform sampling on a 2D-sphere.

    Source: https://gist.github.com/andrewbolster/10274979

    Args
    ----
        num (int): Number of vectors to sample (or None if single)

    Returns
    -------
        np.ndarray: The return value. Random Vector of size (num, 3) with norm 1. 
            If num is None returned value will have size (3,)

    """
    if num is not None:
        phi = np.random.uniform(0.0, 2 * np.pi, num)
        cos_theta = np.random.uniform(-1.0, 1.0, num)
    else:
        phi = np.random.uniform(0.0, 2 * np.pi)
        cos_theta = np.random.uniform(-1.0, 1.0)

    theta = np.arccos(cos_theta)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.stack((x, y, z), axis=-1)


def sample_rotation_matrix(num_axis, augment_rotation):
    """Sample rotation matrix along [num_axis] axis and [0, augment_rotation] angle.

    Args
    ----
        num_axis (int): Rotate along how many axis.
        augment_rotation (float): Rotate by how many angle in radians.

    Returns
    -------
        np.ndarray: Sampled rotation matrix of size [3, 3]
    """
    assert num_axis == 1 or num_axis == 3 or num_axis == 0
    if num_axis == 0:
        return np.eye(3)
    angles = np.random.rand(3) * augment_rotation * np.pi
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(angles[0]), -np.sin(angles[0])],
                   [0, np.sin(angles[0]), np.cos(angles[0])]])
    Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                   [0, 1, 0],
                   [-np.sin(angles[1]), 0, np.cos(angles[1])]])
    Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                   [np.sin(angles[2]), np.cos(angles[2]), 0],
                   [0, 0, 1]])
    # R = Rx @ Ry @ Rz
    if num_axis == 1:
        return random.choice([Rx, Ry, Rz])
    return Rx @ Ry @ Rz


def sample_translation_matrix(augment_translation):
    """Sample translation matrix along 3 axis and [0, augment_translation] meter.
    
    Args
    ----
        augment_translation (float): Translate by how many meters.

    Returns
    -------
        np.ndarray: Sampled translation matrix of size [3, 1].
    """
    T = np.random.rand(3) * augment_translation
    return T.reshape(3, 1)


def transform(pts, trans, norms = None):
    """Apply the SE3 transformations to points (and normals).

    trans_pts = trans[:3, :3] @ pts + trans[:3, 3:4]

    Args
    ----
        pts (np.ndarray): Points to be transformed, [num_pts, 3] or [bs, num_pts, 3]
        trans (np.ndarray): The SE3 transformation matrix, [4, 4] or [bs, 4, 4]
        normals (np.ndarray, optional): Associated normal vectors to be transformed, 
            [num_pts, 3] or [bs, num_pts, 3]

    Returns
    -------
        trans_pts (np.ndarray): Transformed points, [num_pts, 3] or [bs, num_pts, 3]
        trans_norms (np.ndarray): Transformed normal vectors, [num_pts, 3] or 
            [bs, num_pts, 3].
    """
    if len(pts.shape) == 3:
        trans_pts = trans[:, :3, :3] @ pts.permute(0, 2, 1) + trans[:, :3, 3:4]
        if norms is not None:
            trans_norms = trans[:,:3,:3] @ norms.permute(0, 2, 1)
            return trans_pts.permute(0, 2, 1), trans_norms.permute(0, 2, 1)
        return trans_pts.permute(0, 2, 1)
    
    trans_pts = trans[:3, :3] @ pts.T + trans[:3, 3:4]
    if norms is not None:
        trans_norms = trans[:3, :3] @ norms.T
        return trans_pts.T, trans_norms.T
    return trans_pts.T


def decompose_trans(trans):
    """Decompose SE3 transformations into R and t.

    Args
    ----
        trans (np.ndarray): The integrated SE3 transformation matrix, [4, 4] 
            or [bs, 4, 4]
    
    Returns
    -------
        R (np.ndarray): rotation matrix, [3, 3] or [bs, 3, 3]
        t (np.ndarray): translation matrix, [3, 1] or [bs, 3, 1]
    """
    if len(trans.shape) == 3:
        return trans[:, :3, :3], trans[:, :3, 3:4]
    return trans[:3, :3], trans[:3, 3:4]


def integrate_trans(R, t):
    """Integrate SE3 transformations from R and t.

    Args
    ----
        R (np.ndarray): Rotation matrix, [3, 3] or [bs, 3, 3]
        t (np.ndarray): Translation matrix, [3, 1] or [bs, 3, 1]
        
    
    Returns
    -------
        trans (np.ndarray): The integrated SE3 transformation matrix, 
            [4, 4] or [bs, 4, 4]
    """
    if len(R.shape) == 3:
        trans = np.eye(4)[None]
        trans[:, :3, :3] = R
        trans[:, :3, 3:4] = t.view([-1, 3, 1])
    else:
        trans = np.eye(4)
        trans[:3, :3] = R
        trans[:3, 3:4] = t.reshape([3, 1])
    return trans


def concatenate_trans(trans1, trans2):
    """Concatenate two SE3 transformations.

    Args:
        trans1 (np.ndarray): First SE3 matrix, [4, 4] or [bs, 4, 4]
        trans2 (np.ndarray): Second SE3 matrix, [4, 4] or [bs, 4, 4]
    
    Returns:
        trans (np.ndarray): The concatenated SE3 matrix, [4, 4] or [bs, 4, 4]
    """
    R1, t1 = decompose_trans(trans1)
    R2, t2 = decompose_trans(trans2)
    R_cat = R1 @ R2
    t_cat = R1 @ t2 + t1
    trans_cat = integrate_trans(R_cat, t_cat)
    return trans_cat
