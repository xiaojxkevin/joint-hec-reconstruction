import os, torch

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import argparse
import yaml
from utils.model_api import run_mast3r
import utils.geometric_util as geomu
from utils.math_op import inv
from utils.calib import compute_As_Bs, solve_hand_eye_se3
from utils.visual import vis_scene
from utils.ba import run_hand_eye_bundle_adjustment


def load_hand_poses(file_path: str, step: int) -> np.ndarray:
    """
    read transformations from hand to base from a TUM file
    """
    data = np.loadtxt(fname=file_path, delimiter=" ", dtype=np.float32)
    num_check = 8
    if data.shape[1] != num_check:
        raise ValueError(f"Each line in the file should contain {num_check} elements.")
    hand_poses = geomu.tum2transformation(data)
    return hand_poses[::step]


def jcr_run(
    config: dict,
):

    hand2base_poses = load_hand_poses(config["hand_path"], step=config["step"])
    base2hand_poses = np.asarray(
        [inv(pose) for pose in hand2base_poses], dtype=np.float64
    )
    K, eye2obj_poses, points3D, points2D, visibility = run_mast3r(
        input_dir=config["img_dir"],
        output_dir=os.path.join(config["exp_dir"], "colmap"),
        model_path=config["model_path"],
        step=config["step"],
        intrinsics=config["K"],
    )
    assert (
        hand2base_poses.shape[0] == eye2obj_poses.shape[0]
    ), f"{hand2base_poses.shape[0]} != {eye2obj_poses.shape[0]}"
    pts, rgb_colors, pts_errors = points3D[:, :3], points3D[:, 3:6], points3D[:, 6]

    As, Bs = compute_As_Bs(eye2obj_poses, hand2base_poses)

    R_eye2hand, t_eye2hand, scale = solve_hand_eye_se3(As, Bs)
    print("<>" * 20)
    print("R_eye2hand:\n", R_eye2hand)
    print("t_eye2hand: ", t_eye2hand)
    print("scale: ", scale)
    T_eye2hand = np.eye(4, dtype=np.float64)
    T_eye2hand[:3, :3] = R_eye2hand
    T_eye2hand[:3, 3] = t_eye2hand
    np.savetxt(
        os.path.join(config["exp_dir"], "init_T_eye2hand.txt"), T_eye2hand, fmt="%.6f"
    )

    pts *= scale
    eye2obj_poses[:, :3, 3] *= scale
    eye2base_poses = hand2base_poses @ T_eye2hand
    obj2eye_poses = np.asarray([inv(pose) for pose in eye2obj_poses], dtype=np.float64)
    idx = 0
    pts_in_base = geomu.transform_pts_np(pts, eye2base_poses[idx] @ obj2eye_poses[idx])

    ############################# for debugging #############################
    # check for obj2base transformation
    print("<>" * 20)
    print("obj2base transformation:")
    for idx in range(config["num_imgs"]):
        print(eye2base_poses[idx] @ obj2eye_poses[idx])
    ############################# for debugging #############################

    # Visualize constructed ptc
    pts_vis = pts_in_base[::]
    pts_color_vis = rgb_colors[::]
    vis_scene(
        eye2base_poses,
        hand2base_poses,
        pts_vis,
        pts_color_vis,
        os.path.join(config["exp_dir"], "init_scene.html"),
    )
    # For debugging: save the initial transformation matrices
    raw_data = {
        "K": K,
        "eye2base": eye2base_poses,
        "hand2base": hand2base_poses,
        "pts_in_base": pts_in_base,
        "pts_colors": rgb_colors,
        "visibility": visibility,
        "points2D": points2D,
    }
    np.savez(os.path.join(config["exp_dir"], "raw.npz"), **raw_data)

    # Run bundle adjustment
    print(
        "----------------------- Run bundle adjustment ----------------------------------"
    )
    optimized_eye2hand, optimized_points = run_hand_eye_bundle_adjustment(
        K, inv(T_eye2hand), base2hand_poses, pts_in_base, points2D, visibility
    )
    np.savetxt(
        os.path.join(config["exp_dir"], "ba_T_eye2hand.txt"),
        optimized_eye2hand,
        fmt="%.6f",
    )
    eye2base_poses = hand2base_poses @ optimized_eye2hand
    pts_vis = optimized_points[::]
    vis_scene(
        eye2base_poses,
        hand2base_poses,
        pts_vis,
        pts_color_vis,
        os.path.join(config["exp_dir"], "ba_scene.html"),
    )

    # All data are numpy ndarray
    final_data = {
        "K": K,
        "eye2base": eye2base_poses,
        "hand2base": hand2base_poses,
        "pts_in_base": optimized_points,
        "pts_colors": rgb_colors,
    }
    np.savez(os.path.join(config["exp_dir"], "final.npz"), **final_data)
    print("<>" * 20)
    print(f"All results are saved.")


def main():
    parser = argparse.ArgumentParser(
        description="Joint Hand-eye Calibration and Reconstruction"
    )
    parser.add_argument("--exp_name", type=str, required=True, help="Experiment name")
    parser.add_argument(
        "--config_path",
        type=str,
        default="./config/calib.yaml",
        help="Path to the config yaml file",
    )
    args = parser.parse_args()
    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)

    # Load data info
    config["hand_path"] = os.path.join(
        config["data_dir"], args.exp_name, "hand_tum.txt"
    )
    config["img_dir"] = os.path.join(config["data_dir"], args.exp_name, "imgs")
    intrinsic_path = os.path.join(config["data_dir"], args.exp_name, "intrinsics.txt")
    config["K"] = (
        np.loadtxt(intrinsic_path, dtype=np.float64)
        if os.path.exists(intrinsic_path)
        else None
    )
    num_imgs = len(os.listdir(config["img_dir"]))
    config["num_imgs"] = (num_imgs - 1) // config["step"] + 1
    print(f"Using {config['num_imgs']} images for calibration.")

    # The folder to store the results of the experiment
    config["exp_dir"] = os.path.join(
        config["out_dir"], f"{args.exp_name}_{config['num_imgs']:02d}"
    )
    os.makedirs(config["exp_dir"], exist_ok=True)
    print("Working on ", config["exp_dir"])

    # Start joint calibration and reconstruction
    jcr_run(config)


if __name__ == "__main__":
    main()
