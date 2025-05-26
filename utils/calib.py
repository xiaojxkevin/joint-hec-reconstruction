import numpy as np
from scipy.spatial.transform import Rotation
from scipy.linalg import svd, lstsq, solve
from utils.math_op import skew, inv


def compute_As_Bs(eye2obj_poses: np.ndarray, hand2base_poses: np.ndarray):
    """
    Returns: As [N, 4, 4] (relative hand motions), Bs [N, 4, 4] (relative eye motions)
    """
    As, Bs = [], []
    for i in range(1, hand2base_poses.shape[0]):
        eye_pair = eye2obj_poses[i - 1], eye2obj_poses[i]
        hand_pair = hand2base_poses[i - 1], hand2base_poses[i]
        As.append(inv(hand_pair[0]) @ hand_pair[1])
        Bs.append(inv(eye_pair[0]) @ eye_pair[1])
    As = np.asarray(As)
    Bs = np.asarray(Bs)
    return As, Bs


def solve_hand_eye_se3(
    As: np.ndarray,
    Bs: np.ndarray,
    use_ransac=True,
    inlier_ratio=0.75,
    error_threshold=1e-2,
):
    """
    Input:
        As: An array of shape (n, 4, 4) containing SE(3) matrices, where each element is a homogeneous transformation A_i.
        Bs: An array of shape (n, 4, 4) containing SE(3) matrices, where each element is a homogeneous transformation B_i.
    Returns:
        R_X: A 3x3 rotation matrix.
        t_X: A 3x1 translation vector.
        lambda_: A scale factor.
    """
    n = As.shape[0]
    assert n > 0, "Number of SE(3) matrices should be greater than 0."
    # Step 1: Compute the rotation component R_X.
    R_X = retrive_rotation(As, Bs)

    # Step 2: Compute the translation component t_X.
    t_X = None
    C, d = [], []
    for i in range(n):
        # Extract the translation vectors from A_i and B_i.
        tA_i = As[i, :3, 3].reshape(3, 1)
        tB_i = Bs[i, :3, 3].reshape(3, 1)
        # Construct the skew-symmetric matrix for tB_i (denoted as tB_i^).
        tB_skew = skew(tB_i)
        # Compute C_i and d_i.
        RA_i = As[i, :3, :3]
        Ci = tB_skew @ R_X.T @ (np.eye(3) - RA_i)
        di = tB_skew @ R_X.T @ tA_i
        C.append(Ci)
        d.append(di.reshape(-1, 1))  # Ensure the vector is a column vector.

    if use_ransac:
        # RANSAC parameters
        max_iters = 100
        error_threshold = 1e-2
        min_samples = 2
        best_inliers = []
        threshold_num = int(inlier_ratio * n)
        max_inliers = 0

        for _ in range(max_iters):
            # Randomly select minimal sample
            sample_indices = np.random.choice(n, size=min_samples, replace=False)
            C_sample = np.vstack([C[i] for i in sample_indices])
            d_sample = np.vstack([d[i] for i in sample_indices])
            # Solve for t_X candidate
            try:
                t_X_candidate, _, _, _ = lstsq(C_sample, d_sample, check_finite=False)
            except np.linalg.LinAlgError:
                continue  # Skip if singular

            lambda_candidate = retrive_scale_factor(
                np.stack([As[i] for i in sample_indices]),
                np.stack([Bs[i] for i in sample_indices]),
                R_X,
                t_X_candidate,
            )
            # Evaluate inliers
            current_inliers = []
            for i in range(n):
                left_side = As[i, :3, 3].reshape(3, 1)
                right_side = (
                    R_X @ (lambda_candidate * Bs[i, :3, 3].reshape(3, 1))
                    + t_X_candidate
                    - As[i, :3, :3] @ t_X_candidate
                )
                error = np.linalg.norm(left_side - right_side)
                if error < error_threshold:
                    current_inliers.append(i)

            if len(current_inliers) > max_inliers:
                max_inliers = len(current_inliers)
                best_inliers = current_inliers

        # Refit using all inliers
        if len(best_inliers) >= threshold_num:
            print(best_inliers)
            C_inliers = np.vstack([C[i] for i in best_inliers])
            d_inliers = np.vstack([d[i] for i in best_inliers])
            t_X, residues, _, s = lstsq(C_inliers, d_inliers, check_finite=False)
    if t_X is None:
        # Use all data to solve for t_X
        # Solve the linear system C * t_X = d
        C_stack = np.vstack(C)
        d_stack = np.vstack(d)
        try:
            t_X, residues, _, s = lstsq(C_stack, d_stack, check_finite=False)
        except np.linalg.LinAlgError:
            raise ValueError("Linear system could not be solved.")
    lambda_ = retrive_scale_factor(As, Bs, R_X, t_X)

    return R_X, t_X.flatten(), lambda_


def retrive_rotation(As: np.ndarray, Bs: np.ndarray):
    n = As.shape[0]
    M = np.zeros((3, 3))
    for i in range(n):
        # Extract the rotation matrices from A_i and B_i.
        RA_i = As[i, :3, :3]
        RB_i = Bs[i, :3, :3]

        # Convert the rotation matrices to rotation vectors (axis-angle representation).
        w_Ai = Rotation.from_matrix(RA_i).as_rotvec()
        w_Bi = Rotation.from_matrix(RB_i).as_rotvec()
        M += np.outer(w_Ai, w_Bi)

    # Solve the orthogonal Procrustes problem using SVD.
    U, _, Vt = svd(M, check_finite=False)
    R_X = U @ Vt  # Ensure the determinant is +1 (if it's a reflection, adjust Vt).
    if np.linalg.det(R_X) < 0:
        assert False, "Reflection detected. Adjusted the sign of the last row of Vt."
        Vt[-1, :] *= -1
        R_X = U @ Vt
    return R_X


def retrive_scale_factor(
    As: np.ndarray, Bs: np.ndarray, R_X: np.ndarray, t_X: np.ndarray
):
    """
    Compute the scale factor lambda based on the provided As, Bs, R_X, and t_X.
    This function is used to compute the scale factor after obtaining R_X and t_X.
    """
    n = As.shape[0]
    lambda_sum = 0.0
    for i in range(n):
        RA_i = As[i, :3, :3]
        tA_i = As[i, :3, 3].reshape(3, 1)
        tB_i = Bs[i, :3, 3].reshape(3, 1)

        numerator = tB_i.T @ R_X.T @ ((RA_i - np.eye(3)) @ t_X + tA_i)
        denominator = np.linalg.norm(tB_i) ** 2
        assert denominator != 0, "Denominator should not be zero"
        lambda_i = numerator.item() / denominator
        lambda_sum += lambda_i

    return lambda_sum / n


# Example usage.
if __name__ == "__main__":
    # Generate example SE(4) matrices (identity matrices).
    n = 10
    As = np.tile(np.eye(4), (n, 1, 1))  # Shape: (n, 4, 4)
    Bs = np.tile(np.eye(4), (n, 1, 1))  # Shape: (n, 4, 4)

    R_X, t_X, lambda_ = solve_hand_eye_se3(As, Bs)
    print("Rotation R_X:\n", R_X)
    print("Translation t_X:", t_X)
    print("Scale lambda:", lambda_)
