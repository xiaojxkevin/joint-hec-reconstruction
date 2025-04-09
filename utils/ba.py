import numpy as np
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

        # Huber threshold parameter (delta)
        self.huber_delta = 1.0

    def objective_function(self, params):
        """
        Compute the residuals for the bundle adjustment.

        Args:
            params: Parameter vector [hand2eye_params, pts3d_in_base]

        Returns:
            Flattened residual vector
        """
        # Extract hand2eye transformation parameters
        hand2eye_params = params[:6]
        hand2eye_mat = vec2transMat(hand2eye_params)

        # Extract 3D points in base frame
        pts3d_base = params[6:].reshape(self.n_points, 3)

        # Compute residuals for each observation
        residuals = []

        for idx in range(self.n_views):
            base2hand = self.base2hand_poses[idx]
            base2eye = hand2eye_mat @ base2hand

            # Get visible 3D points in base frame for this view
            pts3d_indices = self.visibility[idx]["pts3d_indices"]
            pts3d_base_used = pts3d_base[pts3d_indices]

            # Transform points to eye frame
            pts3d_eye = pts3d_base_used @ base2eye[:3, :3].T + base2eye[:3, 3]

            # Project to 2D
            z = pts3d_eye[:, 2]
            pts2d_projected = np.column_stack(
                [
                    self.fx * pts3d_eye[:, 0] / z + self.cx,
                    self.fy * pts3d_eye[:, 1] / z + self.cy,
                ]
            )

            # Compute residual
            pts2d_used = self.pts2d[self.visibility[idx]["pts2d_indices"]]
            view_residuals = (pts2d_projected - pts2d_used).ravel()
            residuals.extend(view_residuals)

        return np.array(residuals)

    def compute_analytical_jacobian(self, params):
        """
        Compute the analytical Jacobian matrix based on the equations in the PDF,
        optimized with numpy vectorization.

        Args:
            params: Parameter vector [hand2eye_params, pts3d_in_base]

        Returns:
            Jacobian matrix as scipy.sparse.csr_matrix
        """
        # Extract hand2eye transformation parameters
        hand2eye_params = params[:6]
        hand2eye_mat = vec2transMat(hand2eye_params)
        R_eh = hand2eye_mat[:3, :3]
        t_eh = hand2eye_mat[:3, 3]

        # Extract 3D points in base frame
        pts3d_base = params[6:].reshape(self.n_points, 3)

        # Pre-allocate arrays for sparse matrix construction
        # Each point in each view contributes 2 rows (x and y) with 6 + 3 columns
        max_entries = self.total_residuals * (6 + 3)
        data = np.zeros(max_entries)
        row_ind = np.zeros(max_entries, dtype=np.int32)
        col_ind = np.zeros(max_entries, dtype=np.int32)
        entry_idx = 0
        residual_idx = 0

        for view_idx in range(self.n_views):
            base2hand = self.base2hand_poses[view_idx]
            R_hjb = base2hand[:3, :3]
            t_hjb = base2hand[:3, 3]
            pts3d_indices = np.array(self.visibility[view_idx]["pts3d_indices"])
            n_points_in_view = len(pts3d_indices)

            if n_points_in_view == 0:
                raise ValueError("No visible points in this view.")

            # Get all visible 3D points for this view
            P_b_all = pts3d_base[pts3d_indices]

            # Transform all points to hand frame (vectorized)
            P_hj_all = P_b_all @ R_hjb.T + t_hjb

            # Transform all points to eye frame (vectorized)
            P_e_all = P_hj_all @ R_eh.T + t_eh

            # Extract components for projection calculations
            x_all = P_e_all[:, 0]
            y_all = P_e_all[:, 1]
            z_all = P_e_all[:, 2]

            J_xi = np.stack(
                [
                    np.stack(
                        [
                            -self.fx * x_all * y_all / z_all**2,
                            self.fx + self.fx * x_all**2 / z_all**2,
                            -self.fx * y_all / z_all,
                            self.fx / z_all,
                            np.zeros_like(z_all),
                            -self.fx * x_all / z_all**2,
                        ],
                        axis=1,
                    ),
                    np.stack(
                        [
                            -self.fy - self.fy * y_all**2 / z_all**2,
                            self.fy * x_all * y_all / z_all**2,
                            self.fy * x_all / z_all,
                            np.zeros_like(z_all),
                            self.fy / z_all,
                            -self.fy * y_all / z_all**2,
                        ],
                        axis=1,
                    ),
                ],
                axis=1,
            )  # shape: (N, 2, 6)

            J_pts = (
                np.stack(
                    [
                        np.stack(
                            [
                                self.fx / z_all,
                                np.zeros_like(z_all),
                                -self.fx * x_all / z_all**2,
                            ],
                            axis=1,
                        ),
                        np.stack(
                            [
                                np.zeros_like(z_all),
                                self.fy / z_all,
                                -self.fy * y_all / z_all**2,
                            ],
                            axis=1,
                        ),
                    ],
                    axis=1,
                )
                @ R_eh
                @ R_hjb
            )  # shape: (N, 2, 3)

            # For hand-eye transformation parameters (first 6 columns)
            for point_idx in range(n_points_in_view):
                for i in range(2):  # 2 rows per point (x and y residuals)
                    for j in range(6):  # 6 hand-eye parameters
                        row_ind[entry_idx] = residual_idx + point_idx * 2 + i
                        col_ind[entry_idx] = j
                        data[entry_idx] = J_xi[point_idx, i, j]
                        entry_idx += 1

            # For 3D point parameters
            for local_idx, global_idx in enumerate(pts3d_indices):
                for i in range(2):  # 2 rows per point (x and y residuals)
                    for j in range(3):  # 3 coordinates per point
                        row_ind[entry_idx] = residual_idx + local_idx * 2 + i
                        col_ind[entry_idx] = (
                            6 + global_idx * 3 + j
                        )  # Offset by 6 for hand-eye params
                        data[entry_idx] = J_pts[local_idx, i, j]
                        entry_idx += 1

            # Update residual index for the next view
            residual_idx += n_points_in_view * 2

        # Trim any unused pre-allocated space
        data = data[:entry_idx]
        row_ind = row_ind[:entry_idx]
        col_ind = col_ind[:entry_idx]

        # Create sparse matrix
        jacobian = sparse.csr_matrix(
            (data, (row_ind, col_ind)), shape=(self.total_residuals, self.total_params)
        )

        return jacobian

    def custom_least_squares(self, max_iterations=50, ftol=1e-5, verbose=True):
        """
        Custom least squares implementation with Huber loss and analytical Jacobian.

        Args:
            max_iterations: Maximum number of iterations
            ftol: Convergence tolerance on function value
            verbose: Whether to print progress information

        Returns:
            Optimized parameters and optimization result information
        """
        # Initial parameters
        hand2eye_params = transMat2Vec(self.hand2eye)
        params = np.concatenate([hand2eye_params, self.pts3d_in_base.ravel()])

        # Previous cost for convergence check
        prev_cost = np.inf

        # Optimization loop
        from tqdm import tqdm

        for iteration in tqdm(range(max_iterations), desc="Optimization Progress"):
            # Compute residuals
            residuals = self.objective_function(params)
            cost = np.sum(residuals**2)

            # Check convergence
            if np.abs(prev_cost - cost) < ftol:
                if verbose:
                    print(
                        f"Converged at iteration {iteration}: cost difference below tolerance"
                    )
                break

            start = time.time()
            # Compute analytical Jacobian
            jacobian = self.compute_analytical_jacobian(params)
            end = time.time()
            print(f"Jacobian computation time: {end - start:.6f} seconds")

            start = time.time()
            # Compute normal equations: J^T @ J @ Δx = -J^T @ r
            H = jacobian.T @ jacobian
            g = -jacobian.T @ residuals
            end = time.time()
            print(f"H computation time: {end - start:.6f} seconds")

            start = time.time()
            # Solve normal equations with regularization for stability
            lambda_reg = 1e-4  # Levenberg-Marquardt damping factor
            delta_params = linalg.spsolve(
                H + lambda_reg * sparse.eye(self.total_params), g
            )
            end = time.time()
            print(f"Solver time: {end - start:.6f} seconds")

            # Update points
            params[6:] += delta_params[6:]
            # update pose
            params[:6] = transMat2Vec(vec2transMat(delta_params[:6]) @ self.hand2eye)

            if verbose and (iteration % 5 == 0 or iteration == max_iterations - 1):
                print(f"Iteration {iteration}: cost = {cost:.6f}")
            prev_cost = cost

        # Final evaluation
        final_residuals = self.objective_function(params)
        final_cost = np.sum(final_residuals**2)

        if verbose:
            print(f"Optimization completed.")
            print(f"Final cost: {final_cost:.6f}")

        result = {
            "x": params,
            "fun": final_residuals,
            "cost": final_cost,
            "success": True,
            "nit": iteration + 1,
        }

        return result

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
        initial_residuals = self.objective_function(initial_params)
        initial_cost = np.sum(initial_residuals**2)
        print(f"Initial cost: {initial_cost:.6f}")

        result = self.custom_least_squares(max_iterations=50, ftol=1e-5, verbose=True)
        optimized_params = result["x"]

        # Extract optimized parameters
        hand2eye_params = optimized_params[:6]
        optimized_hand2eye = vec2transMat(hand2eye_params)
        optimized_eye2hand = inv(optimized_hand2eye)
        optimized_points = optimized_params[6:].reshape(self.n_points, 3)

        # Compute final cost
        final_residuals = self.objective_function(optimized_params)
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
