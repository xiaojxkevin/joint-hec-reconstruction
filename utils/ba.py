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

            # Vectorized assignment for hand-eye parameters (J_xi)
            n_entries_xi = n_points_in_view * 2 * 6
            end_xi = entry_idx + n_entries_xi
            if end_xi > max_entries:
                raise ValueError("Pre-allocated arrays are too small.")

            data[entry_idx:end_xi] = J_xi.reshape(-1)
            row_ind[entry_idx:end_xi] = (
                residual_idx + np.arange(n_points_in_view * 2)
            ).repeat(6)
            col_ind[entry_idx:end_xi] = np.tile(np.arange(6), n_points_in_view * 2)
            entry_idx = end_xi

            # Vectorized assignment for 3D points (J_pts)
            n_entries_pts = n_points_in_view * 2 * 3
            end_pts = entry_idx + n_entries_pts
            if end_pts > max_entries:
                raise ValueError("Pre-allocated arrays are too small.")

            row_ind_pts = (residual_idx + np.arange(n_points_in_view * 2)).repeat(3)
            row_ind[entry_idx:end_pts] = row_ind_pts

            global_indices = pts3d_indices
            j_values_per_point = np.tile(np.arange(3), 2)
            global_indices_expanded = global_indices.repeat(6)
            j_values_all = np.tile(j_values_per_point, n_points_in_view)
            col_ind_pts = 6 + 3 * global_indices_expanded + j_values_all
            col_ind[entry_idx:end_pts] = col_ind_pts

            data[entry_idx:end_pts] = J_pts.reshape(-1)
            entry_idx = end_pts
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

    def custom_least_squares(self, ftol=1e-6, lambda_=1e-2, verbose=True):
        """ """
        # Initial parameters
        hand2eye_params = transMat2Vec(self.hand2eye)
        params = np.concatenate([hand2eye_params, self.pts3d_in_base.ravel()])

        # Previous cost for convergence check
        prev_cost = np.inf

        # Optimization loop
        for iteration in range(self.max_it):
            # Compute residuals
            residuals = self.objective_function(params)
            cost = np.sqrt(np.sum(residuals**2))

            # Check convergence
            if np.abs(prev_cost - cost) < ftol:
                if verbose:
                    print(
                        f"Converged at iteration {iteration}: cost difference below tolerance"
                    )
                break

            # Compute analytical Jacobian
            jacobian = self.compute_analytical_jacobian(params)

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

            if verbose and (iteration % 1 == 0 or iteration == self.max_it - 1):
                print(f"Iteration {iteration}: cost = {cost:.6f}")
            prev_cost = cost

        result = {
            "x": params,
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
        initial_cost = np.sqrt(np.sum(initial_residuals**2))
        print(f"Initial cost: {initial_cost:.6f}")

        result = self.custom_least_squares(ftol=1e-6)
        optimized_params = result["x"]

        # Extract optimized parameters
        hand2eye_params = optimized_params[:6]
        optimized_hand2eye = vec2transMat(hand2eye_params)
        optimized_eye2hand = inv(optimized_hand2eye)
        optimized_points = optimized_params[6:].reshape(self.n_points, 3)

        # Compute final cost
        final_residuals = self.objective_function(optimized_params)
        final_cost = np.sqrt(np.sum(final_residuals**2))
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
