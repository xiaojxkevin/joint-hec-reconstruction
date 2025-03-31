import numpy as np
from utils.calib import compute_As_Bs, solve_hand_eye_se3
from utils.visual import vis_scene
from utils.gen_data import generate_simulated_data


def vis_true_results(hand2base_poses, eye2hand_true):
    eye2base_poses = hand2base_poses @ eye2hand_true
    vis_scene(eye2base_poses, hand2base_poses, output_path="true_results.html")

def compute_translation_error(T_gt, T_est):
    """
    Computes the Euclidean distance between the translation components of two poses.
    
    Parameters:
        T_gt (np.ndarray): Ground truth 4x4 transformation matrix.
        T_est (np.ndarray): Estimated 4x4 transformation matrix.
    
    Returns:
        float: Translation error.
    """
    t_gt = T_gt[0:3, 3]
    t_est = T_est[0:3, 3]
    error = np.linalg.norm(t_gt - t_est)
    return error

def compute_rotation_error(T_gt, T_est):
    """
    Computes the angular difference (in radians) between the rotation parts of two poses.
    
    Parameters:
        T_gt (np.ndarray): Ground truth 4x4 transformation matrix.
        T_est (np.ndarray): Estimated 4x4 transformation matrix.
    
    Returns:
        float: Rotation error in radians.
    """
    R_gt = T_gt[0:3, 0:3]
    R_est = T_est[0:3, 0:3]
    # Relative rotation matrix
    R_diff = R_gt.T @ R_est
    # Compute the angle using the trace method.
    # Clip the value for numerical stability.
    trace_value = np.clip(np.trace(R_diff), -1.0, 3.0)
    angle = np.arccos((trace_value - 1) / 2)
    return angle

def compute_error(T_gt, T_est):
    """
    Computes the translation and rotation errors between two poses.
    
    Parameters:
        T_gt (np.ndarray): Ground truth 4x4 transformation matrix.
        T_est (np.ndarray): Estimated 4x4 transformation matrix.
    
    Returns:
        tuple: Translation and rotation errors.
    """
    t_error = compute_translation_error(T_gt, T_est)
    R_error = compute_rotation_error(T_gt, T_est)
    return t_error, R_error

def test_instance(As, Bs, eye2hand_true, scale_true, use_ransac):
    R_eye2hand, t_eye2hand, scale = solve_hand_eye_se3(As, Bs, use_ransac, inlier_ratio=0.75)
    T_eye2hand = np.eye(4, dtype=np.float64)
    T_eye2hand[:3, :3] = R_eye2hand
    T_eye2hand[:3, 3] = t_eye2hand
    t_error, R_error = compute_error(eye2hand_true, T_eye2hand)
    s_error = np.abs(scale - scale_true)
    return t_error, R_error, s_error


def main():
    # hand2base_poses, eye2obj_poses, eye2hand_true = generate_simulated_data(
    #     num_samples=10, noise_level=0.03
    # )
    # print("True result")
    # print(eye2hand_true)
    # print(hand2base_poses[0], eye2obj_poses[0])
    # vis_true_results(hand2base_poses, eye2hand_true)

    n = 100
    err_ransasc, err = np.zeros((n, 3)), np.zeros((n, 3))
    for i in range(n):
        hand2base_poses, eye2obj_poses, eye2hand_true, scale_true = generate_simulated_data(num_samples=10, 
                                                                                            noise_level=0.02)
        As, Bs = compute_As_Bs(eye2obj_poses, hand2base_poses)

        # Test with RANSAC
        t_error, R_error, s_error = test_instance(
            As, 
            Bs, 
            eye2hand_true, 
            scale_true, 
            use_ransac=True,
        )
        err_ransasc[i] = t_error, R_error, s_error

        # Test without RANSAC
        t_error, R_error, s_error = test_instance(
            As, 
            Bs, 
            eye2hand_true, 
            scale_true, 
            use_ransac=False
        )
        err[i] = t_error, R_error, s_error

    print("Average error:")
    print(f"Translation error (with/without ransac): {np.mean(err_ransasc[:, 0]):.4f}, {np.mean(err[:, 0]):.4f}")
    print(f"Rotation error (with/without ransac): {np.mean(err_ransasc[:, 1]):.4f}, {np.mean(err[:, 1]):.4f}")
    print(f"Scale error (with/without ransac): {np.mean(err_ransasc[:, 2]):.4f}, {np.mean(err[:, 2]):.4f}")


if __name__ == "__main__":
    main()
