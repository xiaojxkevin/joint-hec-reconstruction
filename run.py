import os, torch

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import numpy as np

np.random.seed(2810)
import argparse
import yaml
from utils.model_api import MAST3R_COLMAP
import utils.geometric_util as geomu
from utils.math_op import inv
from utils.calib import EqSolver
from utils.visual import vis_scene
from utils.ba import HandEyeBundleAdjustment


def load_hand_poses(file_path: str, used_indices: np.ndarray) -> np.ndarray:
    """
    read transformations from hand to base from a TUM file
    """
    data = np.loadtxt(fname=file_path, delimiter=" ", dtype=np.float32)
    num_check = 8
    if data.shape[1] != num_check:
        raise ValueError(f"Each line in the file should contain {num_check} elements.")
    hand_poses = geomu.tum2transformation(data)
    return hand_poses[used_indices]


class JointReconstructCalib:
    def __init__(
        self, cfg_path: str, data_dir: str, out_dir: str, num_imgs=-1) -> None:
        with open(cfg_path, "r") as f:
            config = yaml.safe_load(f)
        config["data_dir"] = data_dir
        config["out_dir"] = out_dir

        # T_hand^base
        config["hand_path"] = os.path.join(config["data_dir"], "hand_tum.txt")

        # RGB camera intrinsics
        intrinsic_path = os.path.join(config["data_dir"], "intrinsics.txt")
        if os.path.exists(intrinsic_path):
            config["K"] = np.loadtxt(intrinsic_path, dtype=np.float64)
        else:
            raise NotImplementedError("No intrinsics is provided!")

        # Image configs
        config["img_dir"] = os.path.join(config["data_dir"], "imgs")
        total_num_imgs = len(os.listdir(config["img_dir"]))
        config["num_imgs"] = total_num_imgs if num_imgs == -1 else num_imgs
        if total_num_imgs < config["num_imgs"]:
            raise ValueError(
                f"Not enough images in {config['img_dir']}. Found {total_num_imgs}, but expected {config['num_imgs']}."
            )
        config["used_ids"] = np.sort(
            np.random.choice(
                np.arange(total_num_imgs), size=config["num_imgs"], replace=False
            )
        )
        print(
            f"Using {config['num_imgs']} images for calibration with indices:",
            config["used_ids"],
        )

        # Save folder
        exp_name = data_dir.split("/")[-1]
        self.save_dir = os.path.join(config["out_dir"], exp_name, f"{config['num_imgs']:02d}_imgs")
        os.makedirs(self.save_dir, exist_ok=True)
        self.exp_name = exp_name

        self.cfg = config

    def run(self):
        # Hand
        hand2base_poses = load_hand_poses(
            self.cfg["hand_path"], used_indices=self.cfg["used_ids"]
        )
        base2hand_poses = np.asarray([inv(pose) for pose in hand2base_poses])

        # Camera pose and scene
        sfm_cfg = {
            "input_dir": self.cfg["img_dir"],
            "output_dir": os.path.join(self.save_dir, "colmap"),
            "model_path": self.cfg["model_path"],
            "used_indices": self.cfg["used_ids"],
            "intrinsics": self.cfg["K"],
        }
        sfm = MAST3R_COLMAP(sfm_cfg)
        K, eye2world_poses, points3D, points2D, visibility = sfm.run()
        assert (
            hand2base_poses.shape[0] == eye2world_poses.shape[0]
        ), f"{hand2base_poses.shape[0]} != {eye2world_poses.shape[0]}"
        pts, rgb_colors, pts_errors = points3D[:, :3], points3D[:, 3:6], points3D[:, 6]
        raw_data = {
            "K": K,
            "eye2world": eye2world_poses,
            "pts_in_world": pts,
            "pts_colors": rgb_colors,
            "visibility": visibility,
            "points2D": points2D,
        }
        np.savez(os.path.join(self.save_dir, f"{self.exp_name}_{self.cfg['num_imgs']:02d}_raw"), **raw_data)

        # Solve AX=XB
        eq_solver =  EqSolver(cfg=self.cfg, eye2world_poses=eye2world_poses, hand2base_poses=hand2base_poses)
        R_eye2hand, t_eye2hand, scale = eq_solver.solve()
        print("<>" * 20)
        T_eye2hand = np.eye(4)
        T_eye2hand[:3, :3] = R_eye2hand
        T_eye2hand[:3, 3] = t_eye2hand
        print("T_eye2hand:\n", T_eye2hand)
        print("scale: ", scale)
        np.savetxt(
            os.path.join(self.save_dir, "init_T_eye2hand.txt"),
            T_eye2hand,
            fmt="%.6f")

        # Fix scale and transform points to base frame
        pts *= scale
        eye2world_poses[:, :3, 3] *= scale
        eye2base_poses = hand2base_poses @ T_eye2hand
        world2eye_poses = np.asarray([inv(pose) for pose in eye2world_poses])
        idx = 0
        pts_in_base = geomu.transform_pts_np(
            pts, eye2base_poses[idx] @ world2eye_poses[idx]
        )

        ############################# for debugging #############################
        # check for obj2base transformation
        print("<>" * 20)
        print("obj2base transformation:")
        for idx in range(self.cfg["num_imgs"]):
            print(eye2base_poses[idx] @ world2eye_poses[idx])
        ############################# for debugging #############################

        # Visualize constructed ptc
        pts_vis = pts_in_base[::]
        pts_color_vis = rgb_colors[::]
        vis_scene(
            eye2base_poses,
            hand2base_poses,
            pts_vis,
            pts_color_vis,
            os.path.join(self.save_dir, "init_scene.html"),
        )
        # For debugging: save the initial transformation matrices
        init_data = {
            "K": K,
            "eye2base": eye2base_poses,
            "hand2base": hand2base_poses,
            "pts_in_base": pts_in_base,
            "pts_colors": rgb_colors,
            "visibility": visibility,
            "points2D": points2D,
        }
        np.savez(os.path.join(self.save_dir, f"{self.exp_name}_{self.cfg['num_imgs']:02d}_init"), **init_data)

        # Run bundle adjustment
        print(
            "----------------------- Run bundle adjustment ----------------------------------"
        )
        ba = HandEyeBundleAdjustment(
            K=K,
            hand2eye_pose=inv(T_eye2hand),
            base2hand_poses=base2hand_poses,
            pts3d_in_base=pts_in_base,
            pts2d=points2D,
            visibility=visibility,
            max_it=self.cfg["ba_max_iter"],
            tol=self.cfg["ba_tolerance"],
        )
        optimized_eye2hand, optimized_points = ba.run_bundle_adjustment()
        np.savetxt(
            os.path.join(self.save_dir, "ba_T_eye2hand.txt"),
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
            os.path.join(self.save_dir, "ba_scene.html"),
        )
        final_data = {
            "K": K,
            "eye2base": eye2base_poses,
            "hand2base": hand2base_poses,
            "pts_in_base": optimized_points,
            "pts_colors": rgb_colors,
            "visibility": visibility,
            "points2D": points2D,
        }
        np.savez(os.path.join(self.save_dir, f"{self.exp_name}_{self.cfg['num_imgs']:02d}_final.npz"), **final_data)
        print("<>" * 20)
        print(f"All results are saved.")


def main():
    cfg_path = "config/calib.yaml"
    
    data_dir = "data/easyscene"
    out_dir = "results"
    sys = JointReconstructCalib(
        cfg_path=cfg_path, data_dir=data_dir, out_dir=out_dir
    )
    sys.run()


if __name__ == "__main__":
    main()
