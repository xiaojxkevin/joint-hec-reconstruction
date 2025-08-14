import numpy as np
from scipy.spatial.transform import Rotation
from scipy.linalg import svd, lstsq
from utils.math_op import skew, inv


class ONLineSolver:
    def __init__(
        self, eye2world_poses: np.ndarray, hand2base_poses: np.ndarray
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

    def _construct_coefficient_matrix(self):
        n = len(self.As)
        A_coeff = np.zeros((12 * n, 13))  # 13 = 9 (vec(R_x)) + 3 (t_x) + 1 (λ)

        for i in range(n):
            # Extract rotation and translation components
            R_ai = self.As[i][:3, :3]
            t_ai = self.As[i][:3, 3]
            R_bi = self.Bs[i][:3, :3]
            t_bi = self.Bs[i][:3, 3]

            # Row indices for this pose pair
            row_start = i * 12

            # First 9 rows: [I_9 - R_ai ⊗ R_bi, 0_9x3, 0_9x1]
            kronecker_product = np.kron(R_ai, R_bi)
            A_coeff[row_start : row_start + 9, :9] = np.eye(9) - kronecker_product
            A_coeff[row_start : row_start + 9, 9:12] = 0  # 0_9x3
            A_coeff[row_start : row_start + 9, 12] = 0  # 0_9x1

            # Last 3 rows: [I_3 ⊗ t_bi^T, I_3 - R_ai, -u_ai]
            t_bi_kron = np.kron(np.eye(3), t_bi.reshape(1, 3))
            A_coeff[row_start + 9 : row_start + 12, :9] = t_bi_kron
            A_coeff[row_start + 9 : row_start + 12, 9:12] = np.eye(3) - R_ai
            A_coeff[row_start + 9 : row_start + 12, 12] = -t_ai

        return A_coeff

    def _normalize_rotation(self, R_):
        """Normalize rotation matrix to ensure proper orthogonality"""
        R = R_.copy()
        det = np.linalg.det(R)

        if np.abs(det) < np.finfo(float).eps:
            raise ValueError("Rotation normalization issue: determinant(R) is null")

        # Make R have unit determinant
        scale = np.sign(det) /  np.cbrt(np.abs(det))
        R *= scale

        # Orthogonalize R using SVD
        U, _, Vt = svd(R)
        R = U @ Vt

        # Handle reflection case
        if np.linalg.det(R) < 0:
            D = np.diag([1.0, 1.0, -1.0])
            R = U @ D @ Vt

        return R

    def _solve_linear_system(self, A_coeff):
        """
        Solve the homogeneous linear system using SVD.

        Args:
            A_coeff: Coefficient matrix

        Returns:
            tuple: (R_x, t_x) where R_x is 3x3 rotation matrix and t_x is 3x1 translation
        """
        # Solve homogeneous system A * x = 0 using SVD
        U, S, Vt = svd(A_coeff, full_matrices=True)

        # Solution is the last column of V (last row of Vt)
        solution = Vt[-1, :]

        # Extract components
        R = solution[:9].reshape(3, 3)
        t_x = solution[9:12]
        lambda_scale = solution[12]

        # Orthogonalize rotation matrix
        R_x = self._normalize_rotation(R)
        
        det = np.linalg.det(R)
        s = np.sign(det) / np.cbrt(np.abs(det))
        lambda_scale *= s
        t_x *= s

        return R_x, t_x, lambda_scale

    def run(self) -> np.ndarray:
        """
        Execute the ONLine hand-eye calibration algorithm.

        Returns:
            np.ndarray: 4x4 homogeneous transformation matrix representing hand-eye calibration
        """
        # Construct the coefficient matrix for the linear system
        A_coeff = self._construct_coefficient_matrix()

        # Solve the linear system
        R_x, t_x, scale = self._solve_linear_system(A_coeff)

        # Construct the final transformation matrix
        T_e2h = np.eye(4)
        T_e2h[:3, :3] = R_x.T
        T_e2h[:3, 3] = (-R_x.T @ t_x).ravel()

        return T_e2h, scale
