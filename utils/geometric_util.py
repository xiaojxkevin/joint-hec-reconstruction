import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
import numpy as np


def transform_pts_np(
    points: np.ndarray, transformation_matrix: np.ndarray
) -> np.ndarray:
    """
    Transform points using a transformation matrix.
    """
    assert (
        points.shape[1] == 3
    ), f"Invalid shape {points.shape} for points. Points should be (n, 3)."
    rotMat = transformation_matrix[:3, :3]
    t = transformation_matrix[:3, 3]
    transformed_points = points @ rotMat.T + t
    return transformed_points


def tum2transformation(data: np.ndarray):
    """Convert an array of shape (n,8) into an array of transformation matrices."""
    if data.shape[1] != 8:
        raise ValueError(
            "Input data must have shape (n,8) with columns time, x, y, z, qx, qy, qz, qw"
        )

    transformations = []

    for row in data:
        _, x, y, z, qx, qy, qz, qw = row

        # Convert quaternion to rotation matrix
        rotation = R.from_quat([qx, qy, qz, qw]).as_matrix()

        # Create homogeneous transformation matrix
        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = [x, y, z]

        transformations.append(transform)

    return np.array(transformations, dtype=np.float32)


def matrices_to_tum(
    transformation_matrices: np.ndarray, timestamps: np.ndarray = None
) -> np.ndarray:
    """
    Convert an array of 4x4 transformation matrices to TUM format:
    [timestamp, x, y, z, qx, qy, qz, qw].

    Parameters:
    - transformation_matrices: An array of shape (n, 4, 4)
    - timestamps: Optional array of timestamps of length n

    Returns:
    - An array of shape (n, 8) in TUM format
    """
    n = transformation_matrices.shape[0]
    tum_data = np.zeros((n, 8))  # Initialize array to store TUM format data

    # Extract translations and rotation matrices from each transformation matrix
    translations = transformation_matrices[:, :3, 3]
    rotations = transformation_matrices[:, :3, :3]
    # Convert rotation matrices to quaternions
    quaternions = R.from_matrix(rotations).as_quat()
    tum_data[:, 1:4] = translations  # Set x, y, z
    tum_data[:, 4:] = quaternions  # Set quaternion values
    if timestamps is not None:
        tum_data[:, 0] = timestamps
    else:
        tum_data[:, 0] = np.arange(n)  # Use simple index as timestamp if none provided

    return tum_data
