"""Evaluation utilities."""

import numpy as np


def transformation_residuals(x1, x2, R, t):
    """Compute the pointwise residuals based on the estimated transformation.
    
    Args
    ----
        x1 (np.ndarray): Points of the first point cloud [bs, n, 3]
        x2 (np.ndarray): Points of the second point cloud [bs, n, 3]
        R (np.ndarray): Estimated rotation matrices [bs, 3, 3]
        t (np.ndarray): Estimated translation vectors [bs, 3, 1]

    Returns
    -------
        res (np.ndarray): Pointwise residuals (Euclidean distance) [b, n, 1]
    """
    x2_reconstruct = np.matmul(R, x1.transpose(0, 2, 1)) + t
    res = np.linalg.norm(x2_reconstruct.transpose(0, 2, 1) - x2, axis=2, keepdims=True)
    return res


def rotation_error(R1, R2):
    """Compute rotation error between the estimated and the ground truth rotation matrix. 
    
    $$r_e = \arc\cos((trace(R_{ij}^{T} R_{ij}^{GT}) - 1) / 2)$$
    
    Args
    ----
        R1 (np.ndarray): Estimated rotation matrices [bs, 3, 3] or [3, 3]
        R2 (np.ndarray): Ground truth rotation matrices [bs, 3, 3] or [3, 3]
    
    Returns
    -------
        ae (np.ndarray): Rotation error in angular degrees [bs]
    """
    if R1.ndim == 2:
        R1 = R1[np.newaxis, :]
        R2 = R2[np.newaxis, :]

    R_ = np.matmul(R1.transpose(0, 2, 1), R2)
    e = np.array([(np.trace(R_[i, :, :]) - 1) / 2 for i in range(R_.shape[0])])

    # Clamp the errors to the valid range (otherwise np.arccos() can result in nan)
    e = np.clip(e, -1, 1)
    ae = np.arccos(e)
    ae = 180. * ae / np.pi
    return ae


def translation_error(t1, t2):
    """Compute translation error between the estimated and the ground truth translation vectors.

    Args
    ----
        t1 (np.ndarray): Estimated translation vectors [bs, 3] or [3]
        t2 (np.ndarray): Ground truth translation vectors [bs, 3] or [3]
    
    Returns
    -------
        te (np.ndarray): translation error in meters [bs]
    """
    if t1.ndim == 3:
        t1 = t1.squeeze(-1)
        t2 = t2.squeeze(-1)

    if t1.ndim == 2:
        trans_error = np.linalg.norm(t1 - t2, axis=1)
    elif t1.ndim == 1:
        trans_error = np.linalg.norm(t1 - t2)
    return trans_error


def get_rot_trans_error(trans_est, trans_gt):
    """Get rotation and translation errors."""
    rot_error = rotation_error(trans_est[..., :3, :3], trans_gt[..., :3, :3])
    trans_error = translation_error(trans_est[..., :3, 3], trans_gt[..., :3, 3])
    return rot_error, trans_error


def get_rot_mse_error(rot_est, rot_gt):
    """rotation MSE error used in DCPNet.

    Args
    ----
        rot_est (torch tensor): [bs, 3, 3]
        rot_gt (torch tensor): [bs, 3, 3]
    
    Returns
    -------
        mse_error (float): Mean squared error of the rotation.
    """
    eye = np.tile(np.eye(3), (rot_gt.shape[0], 1, 1))
    diff = eye - np.matmul(rot_est, np.linalg.inv(rot_gt))
    mse_error = np.mean(np.square(diff))
    return mse_error


def get_trans_mse_error(t_est, t_gt):
    """Translation MSE error used in DCPNet

    Args
    ----
        t_est (torch tensor): [bs, 3, 3]
        t_gt (torch tensor): [bs, 3, 3]

    Returns
    -------
        mse_error (float): Mean squared error of the translation.
    """
    t_error = translation_error(t_est, t_gt)
    return (t_error ** 2).mean()
