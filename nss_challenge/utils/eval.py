import math
import numpy as np


def transformation_residuals(x1, x2, R, t):
    """
    Computer the pointwise residuals based on the estimated transformation paramaters
    
    Args:
        x1  (torch array): points of the first point cloud [b,n,3]
        x2  (torch array): points of the second point cloud [b,n,3]
        R   (torch array): estimated rotation matrice [b,3,3]
        t   (torch array): estimated translation vectors [b,3,1]
    Returns:
        res (torch array): pointwise residuals (Eucledean distance) [b,n,1]
    """
    x2_reconstruct = torch.matmul(R, x1.transpose(1, 2)) + t 
    res = torch.norm(x2_reconstruct.transpose(1, 2) - x2, dim=2)
    return res


def rotation_error(R1, R2):
    """
    Torch batch implementation of the rotation error between the estimated and the ground truth rotatiom matrix. 
    Rotation error is defined as r_e = \arccos(\frac{Trace(\mathbf{R}_{ij}^{T}\mathbf{R}_{ij}^{\mathrm{GT}) - 1}{2})
    Args: 
        R1 (torch tensor): Estimated rotation matrices [b,3,3] / [3, 3]
        R2 (torch tensor): Ground truth rotation matrices [b,3,3] / [3, 3]
    Returns:
        ae (torch tensor): Rotation error in angular degreees [b]
    """
    if R1.ndim == 2:
        R1, R2 = R1.unsqueeze(0), R2.unsqueeze(0)
        
    R_ = torch.matmul(R1.transpose(1,2), R2)
    e = torch.stack([(torch.trace(R_[_, :, :]) - 1) / 2 for _ in range(R_.shape[0])], dim=0)

    # Clamp the errors to the valid range (otherwise torch.acos() is nan)
    e = torch.clamp(e, -1, 1, out=None)
    ae = torch.acos(e)
    pi = torch.Tensor([math.pi])
    ae = 180. * ae / pi.to(ae.device).type(ae.dtype)
    return ae


def translation_error(t1, t2):
    """
    Torch batch implementation of the rotation error between the estimated and the ground truth rotatiom matrix. 
    Rotation error is defined as r_e = \arccos(\frac{Trace(\mathbf{R}_{ij}^{T}\mathbf{R}_{ij}^{\mathrm{GT}) - 1}{2})
    Args: 
        t1 (torch tensor): Estimated translation vectors [b,3] / / [3]
        t2 (torch tensor): Ground truth translation vectors [b,3] / / [3]
    Returns:
        te (torch tensor): translation error in meters [b]
    """
    if t1.ndim == 3: 
        t1, t2 = t1.squeeze(-1), t2.squeeze(-1)
    assert t1.ndim <=2

    if t1.ndim == 2:
        trans_error = torch.norm(t1-t2, dim=1)
    elif t1.ndim == 1:
        trans_error = torch.norm(t1-t2)   
    return trans_error


def get_rot_trans_error(trans_est, trans_gt):
    """
    Get rotation and translation errors
    Args: 
        trans_est (torch tensor): [B, 4, 4]
        trans_gt (torch tensor): [B, 4, 4]
    """
    rot_error = rotation_error(trans_est[:,:3,:3], trans_gt[:,:3,:3])
    trans_error = translation_error(trans_est[:,:3,3], trans_gt[:,:3,3])
    return rot_error, trans_error


def get_rot_mse_error(rot_est, rot_gt):
    """rotation MSE error used in DCPNet

    Args:
        rot_est (torch tensor): [B, 3, 3]
        rot_gt (torch tensor): [b, 3, 3]
    """
    eye = torch.eye(3).expand_as(rot_gt).to(rot_gt.device)
    diff = eye - rot_est @ torch.inverse(rot_gt)
    mse_error = (diff **2).mean()  
    return mse_error


def get_trans_mse_error(t_est, t_gt):
    """translation MSE error used in DCPNet

    Args:
        t_est (torch tensor): [B, 3, 3]
        t_gt (torch tensor): [b, 3, 3]
    """
    t_error = translation_error(t_est, t_gt) 
    return (t_error ** 2).mean()


def get_transformation_mse_error(trans_est, trans_gt):
    """MSE error used in DCPNet

    Args:
        trans_est (torch tensor): [B, 4, 4]
        trans_gt (torch tensor): [b, 4, 4]
    """
    r_mse, t_mse = get_rot_mse_error(trans_est[:,:3,:3], trans_gt[:,:3,:3]), get_trans_mse_error(trans_est[:,:3,3], trans_gt[:,:3,3])
    return r_mse, t_mse
