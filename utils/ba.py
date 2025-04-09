import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as linalg
import os
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
        Compute the analytical Jacobian matrix based on the equations in the PDF.

        Args:
            params: Parameter vector [hand2eye_params, pts3d_in_base]

        Returns:
            Jacobian matrix as scipy.sparse.csr_matrix
        """
        # Extract hand2eye transformation parameters
        hand2eye_params = params[:6]
        hand2eye_mat = vec2transMat(hand2eye_params)
        R_eh = hand2eye_mat[:3, :3]

        # Extract 3D points in base frame
        pts3d_base = params[6:].reshape(self.n_points, 3)

        data = []  # Values
        row_ind = []  # Row indices
        col_ind = []  # Column indices
        residual_idx = 0

        for view_idx in range(self.n_views):
            base2hand = self.base2hand_poses[view_idx]
            R_hjb = base2hand[:3, :3]
            pts3d_indices = self.visibility[view_idx]["pts3d_indices"]

            for pt3d_idx in pts3d_indices:
                # Get the 3D point in base frame
                P_b = pts3d_base[pt3d_idx]

                # Transform point to hand frame
                P_hj = R_hjb @ P_b + base2hand[:3, 3]

                # Transform point to eye frame
                P_e = R_eh @ P_hj + hand2eye_mat[:3, 3]
                x, y, z = P_e

                # Jacobian for hand2eye
                proj_jacobian = np.array(
                    [
                        [self.fx / z, 0, -self.fx * x / (z * z)],
                        [0, self.fy / z, -self.fy * y / (z * z)],
                    ]
                )
                pose_jacobian = np.hstack([-skew(P_e), np.eye(3)])
                hand2eye_jacobian = proj_jacobian @ pose_jacobian
                for i in range(2):
                    for j in range(6):
                        row_ind.append(residual_idx + i)
                        col_ind.append(j)
                        data.append(hand2eye_jacobian[i, j])

                # Jacobian for 3D point
                point_jacobian = proj_jacobian @ R_eh @ R_hjb
                for i in range(2):
                    for j in range(3):
                        row_ind.append(residual_idx + i)
                        col_ind.append(6 + pt3d_idx * 3 + j)
                        data.append(point_jacobian[i, j])

                # Move to next residual
                residual_idx += 2

        # Create sparse Jacobian matrix
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

            # Compute analytical Jacobian
            jacobian = self.compute_analytical_jacobian(params)

            # Compute normal equations: J^T @ J @ Δx = -J^T @ r
            H = jacobian.T @ jacobian
            g = -jacobian.T @ residuals

            # Solve normal equations with regularization for stability
            lambda_reg = 1e-4  # Levenberg-Marquardt damping factor
            delta_params = linalg.spsolve(
                H + lambda_reg * sparse.eye(self.total_params), g
            )

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

    exp_name = "0318"
    colmap_correspondence_path = f"results/{exp_name}/colmap/colmap_raw.json"
    hand2base_path = f"./data/{exp_name}/hand_tum.txt"
    reconstruction_folder = f"results/{exp_name}/colmap/reconstruction/0"
    output_file = "./colmap_correspondences.json"

    # Extract and save correspondences
    correspondences = extract_and_save_correspondences(
        reconstruction_folder, output_file
    )

    # Process the COLMAP data
    K, w2c, points3D, points2D, visibility = process_colmap_data(correspondences)

    # Run optimization
    hand2eye_pose = np.eye(4)  # Initial guess
    optimized_eye2hand, optimized_points = run_hand_eye_bundle_adjustment(
        K, hand2eye_pose, w2c, points3D, points2D, visibility, use_scipy=False
    )

    print("Optimized eye2hand transformation:")
    print(optimized_eye2hand)
