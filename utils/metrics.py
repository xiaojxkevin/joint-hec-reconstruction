import numpy as np
from scipy.spatial.transform import Rotation


def compute_translation_error(T_gt, T_est):
    t_gt = T_gt[:3, 3]
    t_est = T_est[:3, 3]
    error = np.linalg.norm(t_gt - t_est)
    return error


def compute_rotation_error(T_gt, T_est):
    R_gt = T_gt[:3, :3]
    R_est = T_est[:3, :3]
    # Relative rotation matrix
    R_diff = R_gt.T @ R_est

    # return np.linalg.norm(R_diff - np.eye(3), ord="fro")

    # Compute the angle using the trace method.
    # Clip the value for numerical stability.
    trace_value = np.clip(np.trace(R_diff), -1.0, 3.0)
    angle = np.arccos((trace_value - 1) / 2)
    return np.degrees(angle)


def compute_error(T_gt, T_est):
    """
    Computes the translation and rotation errors between the ground truth and estimated transformations.

    Parameters:
    T_gt (numpy.ndarray): The ground truth transformation matrix.
    T_est (numpy.ndarray): The estimated transformation matrix.

    Returns:
    tuple: A tuple (translation error, rotation error).
    """

    t_error = compute_translation_error(T_gt, T_est)
    R_error = compute_rotation_error(T_gt, T_est)
    return t_error, R_error
