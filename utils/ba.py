import numpy as np
import scipy
import scipy.sparse as sparse
import scipy.sparse.linalg as linalg
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
        max_it: int = 10,
    ):
        """
        Initialize the bundle adjustment solver.

        Args:
            K: Camera intrinsic matrix (3x3)
            hand2eye_pose: Initial hand-to-eye transformation matrix (4x4)
            base2hand_poses: Base-to-hand transformation matrices for each view (N x 4x4)
            pts3d_in_base: 3D points in base frame (M x 3)
            pts2d: 2D image points (K x 2)
            visibility: Dictionary containing visibility information for each view
        """
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

        # self.huber_delta = 1.0  # Huber loss threshold
        # Max iterations for optimization
        self.max_it = max_it

    def compute_residuals_and_jacobian(self, params, compute_jacobian=True):
        hand2eye_params = params[:6]
        hand2eye_mat = vec2transMat(hand2eye_params)  # (4, 4)
        R_eh = hand2eye_mat[:3, :3]
        t_eh = hand2eye_mat[:3, 3]
        pts3d_base = params[6:].reshape(-1, 3)  # (n, 3)
        total_residuals = self.total_residuals  # K
        max_entries = total_residuals * (6 + 3)
        data = np.zeros(max_entries)  # (9K,)
        row_ids = np.zeros(max_entries, dtype=np.int32)  # (9K,)
        col_ids = np.zeros(max_entries, dtype=np.int32)  # (9K,)
        entry_idx = 0
        residual_idx = 0
        residuals = np.zeros(total_residuals)  # (K,)
        jacobian = None

        for view_idx in range(self.n_views):
            base2hand: np.ndarray = self.base2hand_poses[view_idx]  # (4, 4)
            R_hjb = base2hand[:3, :3]
            t_hjb = base2hand[:3, 3]
            vis_info = self.visibility[view_idx]

            # Find points can be viewd in this view
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

            # Project points to image plane
            pts2d_projected = np.column_stack(
                [
                    self.fx * x / z + self.cx,
                    self.fy * y / z + self.cy,
                ]
            )  # (n, 2)
            pts2d_used = self.pts2d[vis_info["pts2d_indices"]]
            residuals[residual_idx : residual_idx + 2 * n_points_in_view] = (
                pts2d_projected - pts2d_used
            ).ravel()

            if not compute_jacobian:
                continue

            # Jacobian for pose
            fx, fy = self.fx, self.fy
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

            # Jacobian for points
            J_pts_part1 = np.stack([fx / z, np.zeros_like(z), -fx * x / z**2], axis=1)
            J_pts_part2 = np.stack([np.zeros_like(z), fy / z, -fy * y / z**2], axis=1)
            J_pts_base = np.stack([J_pts_part1, J_pts_part2], axis=1)
            J_pts = J_pts_base @ R_eh @ R_hjb  # (n, 2, 3)

            # Fill in all Jacobian for poses (the first 6 columns)
            n_entries_xi = n_points_in_view * 2 * 6
            end_xi = entry_idx + n_entries_xi
            data[entry_idx:end_xi] = J_xi.ravel()
            row_ids[entry_idx:end_xi] = np.repeat(
                residual_idx + np.arange(2 * n_points_in_view), 6
            )
            col_ids[entry_idx:end_xi] = np.tile(np.arange(6), 2 * n_points_in_view)
            entry_idx = end_xi

            # Fill in all Jacobian for points (only one in each row block)
            n_entries_pts = n_points_in_view * 2 * 3
            end_pts = entry_idx + n_entries_pts
            data[entry_idx:end_pts] = J_pts.ravel()
            row_ids[entry_idx:end_pts] = np.repeat(
                residual_idx + np.arange(2 * n_points_in_view), 3
            )
            col_ids[entry_idx:end_pts] = (
                6
                + 3 * np.repeat(pts3d_indices, 6)
                + np.tile([0, 1, 2, 0, 1, 2], n_points_in_view)
            )
            entry_idx = end_pts

            residual_idx += 2 * n_points_in_view

        if compute_jacobian:
            assert (
                entry_idx == max_entries
            ), f"Entry index mismatch: {entry_idx} != {max_entries}"

            # Define sparse Jacobian matrix
            jacobian = sparse.csr_matrix(
                (data[:entry_idx], (row_ids[:entry_idx], col_ids[:entry_idx])),
                shape=(self.total_residuals, self.total_params),
            )

        return residuals, jacobian

    def custom_least_squares(self, ftol=1e-6, lambda_=1e-2, verbose=True):
        """ """
        # Initial parameters
        hand2eye_params = transMat2Vec(self.hand2eye)
        params = np.concatenate([hand2eye_params, self.pts3d_in_base.ravel()])

        # Previous cost for convergence check
        prev_cost = np.inf

        # Optimization loop
        for iteration in range(1, self.max_it + 1):
            # Compute residuals
            residuals, jacobian = self.compute_residuals_and_jacobian(params)
            cost = np.sum(residuals**2)

            # Check convergence
            if np.abs(prev_cost - cost) < ftol:
                if verbose:
                    print(
                        f"Converged at iteration {iteration}: cost difference below tolerance"
                    )
                break

            # Compute normal equations: J^T @ J @ Δx = -J^T @ r
            H: sparse.csc_matrix = jacobian.T @ jacobian
            g: np.ndarray = -jacobian.T @ residuals

            # Solve the equation
            delta_params = linalg.spsolve(
                H + lambda_ * sparse.eye(self.total_params), g
            )
            lambda_ = max(1e-6, lambda_ * 0.5)
            # TODO: use schur elimination
            # B = H[:6, :6] + np.eye(6) * lambda_reg
            # E = H[:6, 6:]
            # C = H[6:, 6:].tocsc() + sparse.eye(self.n_points * 3) * lambda_reg
            # g1 = g[:6]
            # g2 = g[6:]
            # inv_C = linalg.inv(C)
            # delta_pose = scipy.linalg.solve(B - E @ inv_C @ E.T, g1 - E @ inv_C @ g2)
            # delta_pts = linalg.spsolve(C, g2 - E.T @ delta_pose)

            # Update points
            params[6:] += delta_params[6:]
            # update pose
            params[:6] = transMat2Vec(
                vec2transMat(delta_params[:6]) @ vec2transMat(params[:6])
            )

            if verbose:
                print(f"Iteration {iteration}: cost = {cost:.6f}")
            prev_cost = cost

        return params

    def run_bundle_adjustment(self):
        """
        Run the bundle adjustment optimization.

        Returns:
            Optimized hand2eye transformation and 3D points
        """
        # Initial parameter vector
        hand2eye_params = transMat2Vec(self.hand2eye)
        initial_params = np.concatenate([hand2eye_params, self.pts3d_in_base.ravel()])

        # Compute initial cost
        initial_residuals, _ = self.compute_residuals_and_jacobian(
            initial_params, False
        )
        initial_cost = np.sum(initial_residuals**2)
        print(f"Initial cost: {initial_cost:.6f}")

        optimized_params = self.custom_least_squares(ftol=1e-6)

        # Extract optimized parameters
        hand2eye_params = optimized_params[:6]
        optimized_hand2eye = vec2transMat(hand2eye_params)
        optimized_eye2hand = inv(optimized_hand2eye)
        optimized_points = optimized_params[6:].reshape(self.n_points, 3)

        # Compute final cost
        final_residuals, _ = self.compute_residuals_and_jacobian(
            optimized_params, False
        )
        final_cost = np.sum(final_residuals**2)
        print(f"Final cost: {final_cost:.6f}")

        return optimized_eye2hand, optimized_points


def run_hand_eye_bundle_adjustment(
    K: np.ndarray,
    hand2eye_pose: np.ndarray,
    base2hand_poses: np.ndarray,
    pts3d_in_base: np.ndarray,
    pts2d: np.ndarray,
    visibility: np.ndarray,
):
    """
    Run hand-eye bundle adjustment with analytical Jacobian and Huber loss.

    Args:
        K: Camera intrinsic matrix (3x3)
        hand2eye_pose: Initial hand-to-eye transformation matrix (4x4)
        base2hand_poses: Base-to-hand transformation matrices for each view (N x 4x4)
        pts3d_in_base: 3D points in base frame (M x 3)
        pts2d: 2D image points (K x 2)
        visibility: Dictionary containing visibility information for each view
        use_scipy: Whether to use scipy's least_squares or custom implementation

    Returns:
        Optimized eye2hand transformation and 3D points
    """
    # Run bundle adjustment
    ba = HandEyeBundleAdjustment(
        K, hand2eye_pose, base2hand_poses, pts3d_in_base, pts2d, visibility
    )
    optimized_eye2hand, optimized_points = ba.run_bundle_adjustment()

    return optimized_eye2hand, optimized_points


if __name__ == "__main__":
    from utils.colmap_utils import extract_and_save_correspondences, process_colmap_data

    exp_name = "000_10"
    colmap_correspondence_path = (
        f"results/no_chessboard/{exp_name}/colmap/colmap_raw.json"
    )
    hand2base_path = f"./data/no_chessboard/{exp_name}/hand_tum.txt"
    reconstruction_folder = f"results/no_chessboard/{exp_name}/colmap/reconstruction/0"
    output_file = None

    # Extract and save correspondences
    correspondences = extract_and_save_correspondences(
        reconstruction_folder, output_file
    )

    # Process the COLMAP data
    K, w2c, points3D, points2D, visibility = process_colmap_data(correspondences)

    # Run optimization
    hand2eye_pose = np.eye(4)  # Initial guess
    optimized_eye2hand, optimized_points = run_hand_eye_bundle_adjustment(
        K, hand2eye_pose, w2c, points3D, points2D, visibility
    )

    print("Optimized eye2hand transformation:")
    print(optimized_eye2hand)
