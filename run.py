import os, torch

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import argparse
from utils.model_api import run_mast3r
import utils.geometric_util as geomu
from utils.math_op import inv
from utils.calib import compute_As_Bs, solve_hand_eye_se3
from utils.visual import vis_scene
from utils.ba import run_hand_eye_bundle_adjustment

def load_hand_poses(file_path: str, num: int) -> np.ndarray:
    """
    read transformations from hand to base from a TUM file
    """
    data = np.loadtxt(fname=file_path, delimiter=" ", dtype=np.float32)
    num_check = 8
    if data.shape[1] != num_check:
        raise ValueError(f"Each line in the file should contain {num_check} elements.")
    hand_poses = geomu.tum2transformation(data)
    return hand_poses[:num]


def jcr_run(
    exp_name: str,
    eef_path: str,
    img_dir: str,
    num_imgs: int,
    save_dir: str,
    model_path: str,
):

    hand2base_poses = load_hand_poses(eef_path, num=num_imgs)
    # hand2base_poses = np.asarray([inv(pose) for pose in hand2base_poses], dtype=np.float64)
    K, eye2obj_poses, points3D, points2D, visibility = run_mast3r(
        input_dir=img_dir,
        output_dir=os.path.join(save_dir, "colmap"),
        model_path=model_path,
        num_imgs=num_imgs,
    )
    assert hand2base_poses.shape[0] == eye2obj_poses.shape[0], f"{hand2base_poses.shape[0]} != {eye2obj_poses.shape[0]}"
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
    np.savetxt(f"{save_dir}/init_T_eye2hand.txt", T_eye2hand, fmt="%.6f")

    pts *= scale
    eye2obj_poses[:, :3, 3] *= scale
    eye2base_poses = hand2base_poses @ T_eye2hand
    obj2eye_poses = np.asarray([inv(pose) for pose in eye2obj_poses], dtype=np.float64)
    idx = 0
    pts_in_base = geomu.transform_pts_np(pts, eye2base_poses[idx] @ obj2eye_poses[idx])
    # np.savetxt(f"{save_dir}/eye2base_poses.txt", geomu.matrices_to_tum(eye2base_poses), fmt="%.6f")

    ############################# for debugging #############################
    # check for obj2base transformation
    print("<>" * 20)
    print("obj2base transformation:")
    for idx in range(num_imgs):
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
        f"{save_dir}/init_{exp_name}_{num_imgs}.html",
    )

    # Run bundle adjustment
    print("----------------------- Run bundle adjustment ----------------------------------")
    optimized_eye2hand, optimized_points = run_hand_eye_bundle_adjustment(
        K, inv(T_eye2hand), hand2base_poses, pts_in_base, points2D, visibility
    )
    eye2base_poses = hand2base_poses @ optimized_eye2hand
    pts_vis = optimized_points[::]
    vis_scene(
        eye2base_poses,
        hand2base_poses,
        pts_vis,
        pts_color_vis,
        f"{save_dir}/init_{exp_name}_{num_imgs}.html",
    )

    ############################# for debugging #############################
    # check for obj2base transformation
    print("<>" * 20)
    print("obj2base transformation:")
    for idx in range(num_imgs):
        print(eye2base_poses[idx] @ obj2eye_poses[idx])
    ############################# for debugging #############################

    tensors_to_save = {
        # "K": K_opt,
        "eye2base": eye2base_poses,
        "hand2base": hand2base_poses,
        "pts_in_base": optimized_points,
        "pts_colors": rgb_colors,
    }
    saving_loc = os.path.join(save_dir, f"{exp_name}_{num_imgs}.pth")
    torch.save(tensors_to_save, saving_loc)
    print("<>" * 20)
    print(f"Results saved at {saving_loc}")


def main():
    parser = argparse.ArgumentParser(
        description="Joint Hand-eye Calibration and Reconstruction"
    )
    parser.add_argument("--exp_name", type=str, required=True, help="Experiment name")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./results",
        help="Directory to save the results",
    )
    args = parser.parse_args()

    save_dir = f"{args.out_dir}/{args.exp_name}"
    if not os.path.exists(save_dir):
        print(f"making {save_dir}")
        os.makedirs(save_dir)

    eef_path = f"./data/{args.exp_name}/hand_tum.txt"
    img_dir = f"./data/{args.exp_name}/imgs"
    model_path = "./mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
    num_imgs = 0
    for filename in os.listdir(img_dir):
        ext = os.path.splitext(filename)[1].lower()
        if ext in {".png", ".jpg", ".jpeg"}:
            num_imgs += 1
    # num_imgs = 2

    saving_loc = os.path.join(save_dir, f"{args.exp_name}_{num_imgs}.pth")
    print("Working on", saving_loc)
    jcr_run(
        exp_name=args.exp_name,
        eef_path=eef_path,
        img_dir=img_dir,
        num_imgs=num_imgs,
        save_dir=save_dir,
        model_path=model_path,
    )


if __name__ == "__main__":
    main()
