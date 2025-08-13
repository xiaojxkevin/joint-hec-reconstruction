import numpy as np
import scipy
import scipy.sparse as sparse
import logging
import time
from typing import Dict, List, Tuple, Any
from utils.math_op import inv, skew, transMat2Vec, vec2transMat


class HandEyeBundleAdjustment:
    """
    Bundle Adjustment implementation for hand-eye calibration with analytical Jacobian
    computation and Huber robust cost function.
    """

    def __init__(
        self,
        K: np.ndarray,
        hand2eye_pose: np.ndarray,
        base2hand_poses: np.ndarray,
        pts3d_in_base: np.ndarray,
        pts2d: np.ndarray,
        visibility: np.ndarray,
        cfg: dict,
    ):
        """ """
        self.fx, self.fy, self.cx, self.cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        self.hand2eye = hand2eye_pose
        self.base2hand_poses = base2hand_poses
        self.pts3d_in_base = pts3d_in_base
        self.pts2d = pts2d
        self.visibility = visibility

        self.n_views = base2hand_poses.shape[0]
        self.n_points = pts3d_in_base.shape[0]

        # Pre-calculate total residuals for matrix dimensions
        self.total_residuals = sum(
            len(self.visibility[idx]["pts2d_indices"]) * 2
            for idx in range(self.n_views)
        )

        # Total parameters: 6 for hand2eye + 3 for each 3D point
        self.total_params = 6 + self.n_points * 3
        self.huber_delta = cfg["ba"]["huber_delta"]
        self.mu = cfg["ba"]["damping_factor"]
        self.max_it = cfg["ba"]["max_iter"]
        self.tol = cfg["ba"]["tolerance"]

        self.logger = logging.getLogger("calibration_run")

    def huber_loss(self, residuals):
        square_res = residuals**2
        mask = square_res <= self.huber_delta
        loss1 = np.sum(square_res[mask])
        loss2 = np.sum(
            2.0 * self.huber_delta * np.abs(residuals[~mask]) - self.huber_delta**2
        )
        return loss1 + loss2

    def compute_weights(self, residuals):
        square_res = residuals**2
        mask = square_res <= self.huber_delta
        weights = np.zeros_like(residuals)
        weights[mask] = 1.0
        weights[~mask] = self.huber_delta / np.abs(residuals[~mask])

        n = len(residuals)
        return sparse.dia_matrix((weights, 0), shape=(n, n))

    def compute_residuals_and_jacobian(self, params, compute_jacobian=True):
        hand2eye_params = params[:6]
        hand2eye_mat = vec2transMat(hand2eye_params)  # (4, 4)
        R_eh = hand2eye_mat[:3, :3]
        t_eh = hand2eye_mat[:3, 3]
        pts3d_base = params[6:].reshape(-1, 3)  # (n, 3)
        total_residuals = self.total_residuals  # 2K

        # Preallocate for residuals
        residuals = np.zeros(total_residuals)  # (2K,)

        # Preallocate for J1
        J1 = np.zeros((total_residuals, 6)) if compute_jacobian else None

        # Preallocate for J2
        max_entries = total_residuals * 3
        J2 = None
        data = np.zeros(max_entries)  # (6K,)
        row_ids = np.zeros(max_entries, dtype=np.int32)  # (6K,)
        col_ids = np.zeros(max_entries, dtype=np.int32)  # (6K,)

        # Two indices
        entry_idx = 0
        residual_idx = 0

        for view_idx in range(self.n_views):
            base2hand: np.ndarray = self.base2hand_poses[view_idx]  # (4, 4)
            R_hjb = base2hand[:3, :3]
            t_hjb = base2hand[:3, 3]
            vis_info = self.visibility[view_idx]

            #################################################################### residuals
            # Find points that can be viewd in the current view
            pts3d_indices = np.array(vis_info["pts3d_indices"])
            n_points_in_view = len(pts3d_indices)
            if n_points_in_view == 0:
                continue
            pts3d_used = pts3d_base[pts3d_indices]

            # Points in the camera frame
            P_hj = pts3d_used @ R_hjb.T + t_hjb
            P_e = P_hj @ R_eh.T + t_eh
            x = P_e[:, 0]  # (n, 1)
            y = P_e[:, 1]
            z = P_e[:, 2]

            # Normalize 2d points
            pts2d_used = self.pts2d[vis_info["pts2d_indices"]]
            mean_pts2d = np.mean(pts2d_used, axis=0)
            std_pts2d = np.std(pts2d_used, axis=0)
            s_pts2d = np.sqrt(2.0 / (np.sum(std_pts2d**2) + 1e-6))
            Q_R = np.eye(2) * s_pts2d
            Q_t = -s_pts2d * mean_pts2d
            tilde_pts2d = pts2d_used @ Q_R.T + Q_t

            # Adjust intrinsics
            fx, fy = self.fx * s_pts2d, self.fy * s_pts2d
            cx = s_pts2d * (self.cx - mean_pts2d[0])
            cy = s_pts2d * (self.cy - mean_pts2d[1])

            # Project points to 2D
            pts2d_projected = np.column_stack(
                [
                    fx * x / z + cx,
                    fy * y / z + cy,
                ]
            )  # (n, 2)
            residuals[residual_idx : residual_idx + 2 * n_points_in_view] = (
                pts2d_projected - tilde_pts2d
            ).ravel()

            if not compute_jacobian:
                residual_idx += 2 * n_points_in_view
                continue

            #################################################################### J1
            J_xi_part1 = np.stack(
                [
                    -fx * x * y / z**2,
                    fx + fx * x**2 / z**2,
                    -fx * y / z,
                    fx / z,
                    np.zeros_like(z),
                    -fx * x / z**2,
                ],
                axis=1,
            )
            J_xi_part2 = np.stack(
                [
                    -fy - fy * y**2 / z**2,
                    fy * x * y / z**2,
                    fy * x / z,
                    np.zeros_like(z),
                    fy / z,
                    -fy * y / z**2,
                ],
                axis=1,
            )
            J_xi = np.stack([J_xi_part1, J_xi_part2], axis=1)  # (n, 2, 6)
            J1[residual_idx : residual_idx + 2 * n_points_in_view] = J_xi.reshape(
                (2 * n_points_in_view, 6)
            )

            #################################################################### J2
            J_pts_part1 = np.stack([fx / z, np.zeros_like(z), -fx * x / z**2], axis=1)
            J_pts_part2 = np.stack([np.zeros_like(z), fy / z, -fy * y / z**2], axis=1)
            J_pts_base = np.stack([J_pts_part1, J_pts_part2], axis=1)
            J_pts = J_pts_base @ R_eh @ R_hjb  # (n, 2, 3)

            # Fill in all Jacobian for points (only one in each row block)
            n_entries_pts = n_points_in_view * 2 * 3
            end_pts = entry_idx + n_entries_pts
            data[entry_idx:end_pts] = J_pts.ravel()
            row_ids[entry_idx:end_pts] = np.repeat(
                residual_idx + np.arange(2 * n_points_in_view), 3
            )
            col_ids[entry_idx:end_pts] = 3 * np.repeat(pts3d_indices, 6) + np.tile(
                [0, 1, 2, 0, 1, 2], n_points_in_view
            )
            entry_idx = end_pts

            residual_idx += 2 * n_points_in_view

        if compute_jacobian:
            assert (
                entry_idx == max_entries
            ), f"Entry index mismatch: {entry_idx} != {max_entries}"

            # Define sparse Jacobian matrix
            J2 = sparse.csr_matrix(
                (data, (row_ids, col_ids)),
                shape=(self.total_residuals, self.total_params - 6),
            )

        return residuals, J1, J2

    def invert_block_diagonal_csc(self, C: sparse.csc_matrix):
        inv_blocks = []

        for k in range(self.n_points):
            # Extract the k-th 3x3 block
            cols = [3 * k, 3 * k + 1, 3 * k + 2]
            block = np.zeros((3, 3))

            for i, col in enumerate(cols):
                start = C.indptr[col]
                end = C.indptr[col + 1]
                data_col = C.data[start:end]
                block[:, i] = data_col

            # Invert the block
            inv_block = scipy.linalg.inv(block, check_finite=False)
            inv_blocks.append(inv_block)

        inv_C = sparse.block_diag(inv_blocks, format="csc")
        return inv_C

    def custom_least_squares(self):
        """ """
        # Initial parameters
        hand2eye_params = transMat2Vec(self.hand2eye)
        params = np.concatenate([hand2eye_params, self.pts3d_in_base.ravel()])
        prev_cost = np.inf

        # Optimization loop
        for iteration in range(1, self.max_it + 1):
            # Compute residuals
            residuals, J1, J2 = self.compute_residuals_and_jacobian(params)
            huber_loss = self.huber_loss(residuals)
            self.logger.info("Iteration %d: Huber loss = %.6f", iteration, huber_loss)

            W = self.compute_weights(residuals)
            B: np.ndarray = J1.T @ W @ J1
            for i in range(6):
                B[i, i] += self.mu * B[i, i]
            E: np.ndarray = J1.T @ W @ J2
            C: sparse.csc_matrix = J2.T @ W @ J2
            C += self.mu * sparse.dia_matrix(
                (C.diagonal(), 0), shape=(self.total_params - 6, self.total_params - 6)
            )
            inv_C = self.invert_block_diagonal_csc(C)
            g1: np.ndarray = -J1.T @ W @ residuals
            g2: np.ndarray = -J2.T @ W @ residuals

            # Solve the equation
            delta_pose = scipy.linalg.solve(
                B - E @ inv_C @ E.T,
                g1 - E @ inv_C @ g2,
                check_finite=False,
                assume_a="hermitian",
            )
            delta_points = inv_C @ (g2 - E.T @ delta_pose)

            # update damping factor
            if iteration > 1:
                cost = residuals.reshape((-1, 1)).T @ W @ residuals.reshape((-1, 1))
                if cost < prev_cost:
                    self.mu *= 0.1
                    prev_cost = cost
                else:
                    self.logger.warning(
                        "!!!!! Iteration %d: Cost did not decrease, increasing mu and do not update",
                        iteration,
                    )
                    self.mu *= 10.0
                    continue

            # Update points
            params[6:] += delta_points
            # update pose
            params[:6] = transMat2Vec(
                vec2transMat(delta_pose) @ vec2transMat(params[:6])
            )

            if np.linalg.norm(delta_pose) < self.tol:
                self.logger.info(
                    "Converged at iteration %d: cost difference below tolerance",
                    iteration,
                )
                break

        return params

    def run_bundle_adjustment(self):
        """
        Run the bundle adjustment optimization.

        Returns:
            Optimized hand2eye transformation and 3D points
        """
        # Initial parameter vector
        hand2eye_params = transMat2Vec(self.hand2eye)

        optimized_params = self.custom_least_squares()

        # Extract optimized parameters
        hand2eye_params = optimized_params[:6]
        optimized_hand2eye = vec2transMat(hand2eye_params)
        optimized_eye2hand = inv(optimized_hand2eye)
        optimized_points = optimized_params[6:].reshape(self.n_points, 3)

        return optimized_eye2hand, optimized_points


if __name__ == "__main__":
    pass
