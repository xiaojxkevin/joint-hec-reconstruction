import numpy as np
import scipy.sparse as sparse
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R
import os

from utils.math_op import inv, transMat2Vec, vec2transMat

class HandEyeBundleAdjustment:
    """
    """
    
    def __init__(self, K:np.ndarray, 
                 hand2eye_pose:np.ndarray, 
                 base2hand_poses:np.ndarray, 
                 pts3d_in_base:np.ndarray, 
                 pts2d:np.ndarray, 
                 visibility:np.ndarray):
        """
        """
        # self.K = K
        self.fx, self.fy, self.cx, self.cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        self.hand2eye = hand2eye_pose
        self.base2hand_poses = base2hand_poses
        self.pts3d_in_base = pts3d_in_base
        self.pts2d = pts2d
        self.visibility = visibility

        self.n_views = base2hand_poses.shape[0]
        self.n_points = pts3d_in_base.shape[0]
    
    
    def objective_function(self, params):
        """
        """
        # Extract eye2hand transformation parameters
        hand2eye_params = params[:6]
        hand2eye_mat = vec2transMat(hand2eye_params)
        
        # Extract 3D points in base frame
        pts3d_base = params[6:].reshape(self.n_points, 3)
        
        # Compute residuals for each observation
        residuals = []
        
        for idx in range(self.n_views):
            base2hand = self.base2hand_poses[idx]
            base2eye = base2hand @ hand2eye_mat

            pts3d_base_used = pts3d_base[self.visibility[idx]["pts3d_indices"]]
            pts3d_eye = pts3d_base_used @ base2eye[:3, :3].T + base2eye[:3, 3]
            
            # Project to 2D
            pts3d_eye_normalized = pts3d_eye / pts3d_eye[:, 2][:, None]
            pts2d_projected = np.array([self.fx * pts3d_eye_normalized[:, 0] + self.cx,
                                        self.fy * pts3d_eye_normalized[:, 1] + self.cy]).T

            # Compute residual
            pts2d_used = self.pts2d[self.visibility[idx]["pts2d_indices"]]
            residual = pts2d_projected - pts2d_used
            residuals.extend(residual)
            
        return np.array(residuals).ravel()
    
    def run_bundle_adjustment(self):
        """
        """
        # Initial parameter vector
        # intrinsics_params = np.array([self.fx, self.fy])
        hand2eye_params = transMat2Vec(self.hand2eye)
        initial_params = np.concatenate([ hand2eye_params, 
                                         self.pts3d_in_base.ravel()])
        
        # Compute sparsity pattern
        # sparsity = self.jacobian_sparsity()
        
        # Run the optimization
        print("Starting bundle adjustment optimization...")
        result = least_squares(
            self.objective_function,
            initial_params,
            verbose=2,
            loss="huber",
            method='trf',
            ftol=1e-4,
        )
        
        # Extract optimized parameters
        optimized_params = result.x
        # fx, fy = optimized_params[:2]
        hand2eye_params = optimized_params[:6]
        optimized_eye2hand = inv(vec2transMat(hand2eye_params))
        optimized_points = optimized_params[6:].reshape(self.n_points, 3)
        
        print("Bundle adjustment completed.")
        print(f"Initial cost: {np.sum(self.objective_function(initial_params)**2)}")
        print(f"Final cost: {np.sum(result.fun**2)}")
        
        return optimized_eye2hand, optimized_points


def run_hand_eye_bundle_adjustment(K:np.ndarray, 
                                   hand2eye_pose:np.ndarray,
                                   base2hand_poses:np.ndarray,
                                   pts3d_in_base:np.ndarray,
                                   pts2d:np.ndarray,
                                   visibility:np.ndarray):
    """
    """

    # Run bundle adjustment
    ba = HandEyeBundleAdjustment(K,
                                 hand2eye_pose,
                                 base2hand_poses,
                                 pts3d_in_base,
                                 pts2d,
                                 visibility)
    optimized_eye2hand, optimized_points = ba.run_bundle_adjustment()
    # K_opt = np.array([[fx, 0, K[0, 2]], [0, fy, K[1, 2]], [0, 0, 1]])
    
    return optimized_eye2hand, optimized_points


if __name__ == "__main__":
    from utils.colmap_utils import extract_and_save_correspondences, process_colmap_data
    exp_name = "0318"
    colmap_correspondence_path = f"results/{exp_name}/colmap/colmap_raw.json"
    hand2base_path = f"./data/{exp_name}/hand_tum.txt"
    reconstruction_folder = f"results/{exp_name}/colmap/reconstruction/0"
    output_file = "./colmap_correspondences.json"
    
    # Extract and save correspondences
    correspondences = extract_and_save_correspondences(reconstruction_folder, output_file)
    
    # Process the COLMAP data
    K, w2c, points3D, points2D, visibility = process_colmap_data(correspondences)


    
