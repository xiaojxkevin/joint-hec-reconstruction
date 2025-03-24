import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
import numpy as np


def transform_pts_np(
    points: np.ndarray, transformation_matrix: np.ndarray
) -> np.ndarray:
    """
    Apply a 4x4 transformation matrix to an array of 3D points.

    [IN] points: numpy array of shape (n, 3) representing the point coordinates [x, y, z]
    [IN] transformation_matrix: 4x4 numpy array representing the transformation matrix
    [OUT] transformed_points: numpy array of shape (n, 3) representing the transformed coordinates [x', y', z']
    """
    assert points.shape[1] == 3, "Invalid shape for points. Points should be (n, 3)."
    # Convert points to homogeneous coordinates by appending a column of ones.
    ones = np.ones((points.shape[0], 1))
    points_homogeneous = np.hstack((points, ones))

    # Apply the transformation using matrix multiplication.
    # Note: We use transformation_matrix.T so that each point is transformed correctly.
    transformed_points_homogeneous = points_homogeneous.dot(transformation_matrix.T)

    # Convert back to Cartesian coordinates by taking the first three columns.
    transformed_points = transformed_points_homogeneous[:, :3]

    return transformed_points


def transform_pts_tor(points: torch.Tensor, matrix: torch.Tensor):
    """
    Applies a transformation to a set of 3D points.
    [IN] points: A torch tensor of size (n, 3) representing n 3D points.
    [IN] matrix: A torch tensor of size (4, 4) representing the transformation matrix.
    [OUT] A torch tensor of size (n, 3) of transformed 3D points.
    """
    # Check if the inputs are torch tensors
    if not isinstance(points, torch.Tensor) or not isinstance(matrix, torch.Tensor):
        raise ValueError("Both points and matrix must be torch.Tensor objects.")

    # Check the shape of the points and the matrix
    if points.shape[1] != 3 or matrix.shape != (4, 4):
        raise ValueError(
            "Invalid shape for points or matrix. Points should be (n, 3) and matrix should be (4, 4)."
        )

    # Add an extra dimension of ones to the points tensor to make it compatible with the transformation matrix
    ones = torch.ones(points.shape[0], 1, dtype=points.dtype, device=points.device)
    points_homogeneous = torch.cat(
        [points, ones], dim=1
    )  # Convert points to homogeneous coordinates

    # Apply the transformation matrix to the points
    transformed_points_homogeneous = torch.mm(
        points_homogeneous, matrix.t()
    )  # Multiply by the transpose of the matrix

    # Convert back from homogeneous coordinates by dropping the last dimension
    transformed_points = transformed_points_homogeneous[:, :3]

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
