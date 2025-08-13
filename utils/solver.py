import numpy as np
import logging
from scipy.spatial.transform import Rotation
from scipy.linalg import svd, lstsq
from utils.math_op import skew, inv


class AXXBSolver:
    def __init__(
        self, cfg: dict, eye2world_poses: np.ndarray, hand2base_poses: np.ndarray
    ) -> None:
        assert (
            eye2world_poses.shape[0] == hand2base_poses.shape[0]
        ), "The number of poses in eye2world and hand2base should be the same."
        As, Bs = [], []
        for i in range(1, hand2base_poses.shape[0]):
            eye_pair = eye2world_poses[i - 1], eye2world_poses[i]
            hand_pair = hand2base_poses[i - 1], hand2base_poses[i]
            As.append(inv(eye_pair[0]) @ eye_pair[1])
            Bs.append(inv(hand_pair[0]) @ hand_pair[1])
        self.As = np.asarray(As)
        self.Bs = np.asarray(Bs)

        self.use_ransac = cfg["ransac"]["use_ransac"]
        self.max_iters = cfg["ransac"].get("max_iters", 100)
        self.inlier_ratio = cfg["ransac"].get("inlier_ratio", 0.75)
        self.error_threshold = cfg["ransac"].get("error_threshold", 1e-2)
        self.min_samples = cfg["ransac"].get("min_samples", 2)
        self.logger = logging.getLogger("hand_eye_calibration")

    def retrive_rotation(self, As: np.ndarray, Bs: np.ndarray) -> np.ndarray:
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
        R_X = U @ Vt
        # Ensure the determinant is +1 (if it's a reflection, adjust Vt).
        # https://github.com/opencv/opencv/blob/4.x/modules/calib3d/src/calibration_handeye.cpp#L542
        if np.linalg.det(R_X) < 0:
            self.logger.warning(
                "Reflection detected. Adjusting the sign of the last row of Vt"
            )
            Vt[-1, :] *= -1
            R_X = U @ Vt
        return R_X

    def retrive_scale_factor(
        self, As: np.ndarray, Bs: np.ndarray, R_X: np.ndarray, t_X: np.ndarray
    ):
        """
        Compute the scale factor lambda based on the provided As, Bs, R_X, and t_X.
        This function is used to compute the scale factor after obtaining R_X and t_X.
        """
        n = As.shape[0]
        lambda_list = []
        for i in range(n):
            RA_i = As[i, :3, :3]
            tA_i = As[i, :3, 3].reshape(3, 1)
            sfm_norm = np.linalg.norm(tA_i)
            uA_i = tA_i / sfm_norm
            tB_i = Bs[i, :3, 3].reshape(3, 1)

            real_norm = uA_i.T @ (R_X @ tB_i + (np.eye(3) - RA_i) @ t_X)
            lambda_i = real_norm.item() / sfm_norm
            lambda_list.append(lambda_i)

        return np.average(lambda_list)

    def run(self):
        As, Bs = self.As, self.Bs
        n = As.shape[0]
        assert n > 0, "Number of SE(3) matrices should be greater than 0."
        # Step 1: Compute the rotation component R_X.
        R_X = self.retrive_rotation(self.As, self.Bs)

        # Step 2: Compute the translation component t_X.
        t_X = None
        C, d = [], []
        for i in range(n):
            tA_i = As[i, :3, 3].reshape(3, 1)
            uA_i = tA_i / np.linalg.norm(tA_i)
            tB_i = Bs[i, :3, 3].reshape(3, 1)
            RA_i = As[i, :3, :3]
            # Compute C_i and d_i.
            uA_skew = skew(uA_i)
            Ci = uA_skew @ (RA_i - np.eye(3))
            di = uA_skew @ R_X @ tB_i
            C.append(Ci)
            d.append(di.reshape(-1, 1))  # Ensure the vector is a column vector.

        if self.use_ransac:
            # RANSAC parameters
            best_inliers = []
            threshold_num = int(self.inlier_ratio * n)
            max_inliers = 0

            for _ in range(self.max_iters):
                # Randomly select minimal sample
                sample_indices = np.random.choice(
                    n, size=self.min_samples, replace=False
                )
                C_sample = np.vstack([C[i] for i in sample_indices])
                d_sample = np.vstack([d[i] for i in sample_indices])
                # Solve for t_X candidate
                try:
                    t_X_candidate, _, _, _ = lstsq(
                        C_sample, d_sample, check_finite=False
                    )
                except np.linalg.LinAlgError:
                    continue  # Skip if singular

                lambda_candidate = self.retrive_scale_factor(
                    np.stack([As[i] for i in sample_indices]),
                    np.stack([Bs[i] for i in sample_indices]),
                    R_X,
                    t_X_candidate,
                )
                # Evaluate inliers
                current_inliers = []
                for i in range(n):
                    left_side = As[i, :3, :3] @ t_X_candidate + lambda_candidate * As[
                        i, :3, 3
                    ].reshape(3, 1)
                    right_side = R_X @ Bs[i, :3, 3].reshape(3, 1) + t_X_candidate
                    error = np.linalg.norm(left_side - right_side)
                    if error < self.error_threshold:
                        current_inliers.append(i)

                if len(current_inliers) > max_inliers:
                    # print(f"there are {len(current_inliers)} inliers")
                    max_inliers = len(current_inliers)
                    best_inliers = current_inliers

            # Refit using all inliers
            if len(best_inliers) >= max(threshold_num, 2):
                self.logger.info("RANSAC found %s inliers.", len(best_inliers))
                C_inliers = np.vstack([C[i] for i in best_inliers])
                d_inliers = np.vstack([d[i] for i in best_inliers])
                t_X, residues, _, s = lstsq(C_inliers, d_inliers, check_finite=False)

        if t_X is None:
            self.logger.info("All data are used to solve for t_X")
            # Use all data to solve for t_X
            # Solve the linear system C * t_X = d
            C_stack = np.vstack(C)
            d_stack = np.vstack(d)
            try:
                t_X, residues, _, s = lstsq(C_stack, d_stack, check_finite=False)
            except np.linalg.LinAlgError:
                raise ValueError("Linear system could not be solved.")
        lambda_ = self.retrive_scale_factor(As, Bs, R_X, t_X)

        R_e2h = R_X.T
        t_e2h = -R_X.T @ t_X.reshape(3, 1)

        return R_e2h, t_e2h.ravel(), lambda_


if __name__ == "__main__":
    pass
