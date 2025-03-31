import numpy as np
import random
from utils.math_op import inv


def random_se3(max_translation=1.0, max_rotation_deg=180):
    """
    Generate a random SE(3) transformation matrix (including rotation and translation).

    [IN] max_translation: Maximum translation value in each axis.
    [IN] max_rotation_deg: Maximum rotation in degrees.
    [OUT] T: A 4x4 numpy array representing the SE(3) transformation.
    """
    # Generate a random translation vector
    t = np.random.uniform(-max_translation, max_translation, 3)

    # Generate a random rotation (a random angle around a random axis)
    theta = np.radians(random.uniform(-max_rotation_deg, max_rotation_deg))
    axis = np.random.randn(3)
    axis /= np.linalg.norm(axis)  # Normalize the rotation axis
    K = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
    )
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)

    # Construct the SE(3) matrix
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def generate_simulated_data(num_samples=10, noise_level=0.0):
    """
    Generate simulated data.

    This function defines the ground truth parameters (assumed unknown and to be determined by calibration),
    and then generates simulated poses.

    [IN] num_samples: Number of sample poses to generate.
    [IN] noise_level: Maximum noise level for the transformation.
    [OUT] hand_to_base_list: An array of hand-to-base poses of shape (n, 4, 4).
    [OUT] cam_to_object_list: An array of camera-to-object poses of shape (n, 4, 4).
    [OUT] cam_to_hand_true: The ground truth camera-to-hand transformation (4x4 matrix).
    """
    # ----------------------------
    # Define ground truth parameters (assumed unknown, to be determined by calibration)
    # ----------------------------
    # Ground truth transformation from camera to hand (X to be determined by calibration)
    cam_to_hand_true = random_se3(max_translation=0.1, max_rotation_deg=60)
    # Fixed pose of the object in the base coordinate system
    base_to_object = random_se3(max_translation=1.0, max_rotation_deg=60)
    scale_factor = 5 * np.random.rand() + 1e-2

    # ----------------------------
    # Generate simulated data
    # ----------------------------
    hand_to_base_list = []
    cam_to_object_list = []

    for _ in range(num_samples):
        # 1. Generate a random pose for hand_to_base
        hand_to_base = random_se3(max_translation=0.5, max_rotation_deg=45)
        hand_to_base_list.append(hand_to_base)

        # 2. Compute the pose for cam_to_object (derived from the ground truth parameters)
        cam_to_hand = cam_to_hand_true  # Ground truth value
        cam_to_base = (
            hand_to_base @ cam_to_hand
        )  # Camera pose = hand pose @ cam_to_hand
        cam_to_object = base_to_object @ cam_to_base
        cam_to_object[:3, :3] /= scale_factor

        # Optional: add noise
        if noise_level > 0:
            noise = random_se3(
                max_translation=noise_level, max_rotation_deg=noise_level * 180 / 3.14
            )
            cam_to_object = cam_to_object @ noise

        cam_to_object_list.append(cam_to_object)

    return (
        np.asarray(hand_to_base_list),
        np.asarray(cam_to_object_list),
        cam_to_hand_true,
        scale_factor
    )


if __name__ == "__main__":
    pass