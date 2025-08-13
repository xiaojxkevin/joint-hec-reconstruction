import os, torch

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import argparse
import yaml
import logging
from utils.mast3r_colmap import MAST3R_COLMAP
import utils.util as util_pkg
from utils.math_op import inv
from utils.solver import AXXBSolver
from utils.visual import vis_scene
from utils.ba import HandEyeBundleAdjustment
from utils.logger import LoggerSetup


def load_hand_poses(file_path: str, used_indices: np.ndarray) -> np.ndarray:
    """
    read transformations from hand to base from a TUM file
    """
    data = np.loadtxt(fname=file_path, delimiter=" ", dtype=np.float32)
    num_check = 8
    if data.shape[1] != num_check:
        raise ValueError(f"Each line in the file should contain {num_check} elements.")
    hand_poses = util_pkg.tum2transformation(data)
    return hand_poses[used_indices]


class JointReconstructCalib:
    def __init__(
        self,
        cfg_path: str,
        data_dir: str,
        out_dir: str,
        exp_name="",
        num_imgs=-1,
    ) -> None:
        # initialize random seed
        np.random.seed(2810)
        logger.info("Start the program")
        with open(cfg_path, "r") as f:
            config = yaml.safe_load(f)
        config["data_dir"] = data_dir
        config["out_dir"] = out_dir

        # T_hand^base
        config["hand_path"] = os.path.join(config["data_dir"], "hand_tum.txt")

        # RGB camera intrinsics
        intrinsic_path = os.path.join(config["data_dir"], "intrinsics.txt")
        if os.path.exists(intrinsic_path):
            config["K"] = np.loadtxt(intrinsic_path)
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
        config["used_ids"] = np.arange(config["num_imgs"])
        logger.info(
            "Using %d images for calibration with indices: %s",
            config["num_imgs"],
            config["used_ids"],
        )

        # Save folder
        if not exp_name:
            exp_name = data_dir.split("/")[-1]
        self.save_dir = os.path.join(
            config["out_dir"], exp_name, f"{config['num_imgs']:02d}_imgs"
        )
        os.makedirs(self.save_dir, exist_ok=True)
        self.exp_name = exp_name

        # Save configuration
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
            "hand2base": hand2base_poses,
            "eye2world": eye2world_poses,
            "pts_in_world": pts,
            "pts_colors": rgb_colors,
            "visibility": visibility,
            "points2D": points2D,
        }
        np.savez(
            os.path.join(
                self.save_dir, f"{self.exp_name}_{self.cfg['num_imgs']:02d}_raw"
            ),
            **raw_data,
        )

        # Solve AX=XB
        logger.info("Start solving AX=XB")
        eq_solver = AXXBSolver(
            cfg=self.cfg,
            eye2world_poses=eye2world_poses,
            hand2base_poses=hand2base_poses,
        )
        R_eye2hand, t_eye2hand, scale = eq_solver.run()
        T_eye2hand = np.eye(4)
        T_eye2hand[:3, :3] = R_eye2hand
        T_eye2hand[:3, 3] = t_eye2hand
        logger.info("T_eye2hand:\n%s", T_eye2hand)
        logger.info("scale: %s", scale)
        np.savetxt(
            os.path.join(self.save_dir, "init_T_eye2hand.txt"), T_eye2hand, fmt="%.6f"
        )

        # Fix scale and transform points to base frame
        pts *= scale
        eye2world_poses[:, :3, 3] *= scale
        eye2base_poses = hand2base_poses @ T_eye2hand
        world2eye_poses = np.asarray([inv(pose) for pose in eye2world_poses])
        idx = 0
        pts_in_base = util_pkg.transform_pts_np(
            pts, eye2base_poses[idx] @ world2eye_poses[idx]
        )

        ############################# for debugging #############################
        # check for obj2base transformation
        # print("obj2base transformation:")
        # for idx in range(self.cfg["num_imgs"]):
        #     print(eye2base_poses[idx] @ world2eye_poses[idx])
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
        np.savez(
            os.path.join(
                self.save_dir, f"{self.exp_name}_{self.cfg['num_imgs']:02d}_init"
            ),
            **init_data,
        )

        # Run bundle adjustment
        logger.info("Start Bundle Adjustment")
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
        np.savez(
            os.path.join(
                self.save_dir, f"{self.exp_name}_{self.cfg['num_imgs']:02d}_final.npz"
            ),
            **final_data,
        )
        logger.info("All results are saved to %s\n", self.save_dir)


if __name__ == "__main__":
    # Set up logger
    logger_setup = LoggerSetup(
        name="hand_eye_calibration",
        log_file="calibration_run.log",
        file_level=logging.DEBUG,
        console_level=logging.INFO,
    )
    logger = logger_setup.setup_logger()

    # Set up args for running the script
    parser = argparse.ArgumentParser(
        description="A script to process calibration and data files."
    )
    parser.add_argument(
        "--cfg_path",
        type=str,
        default="config/calib.yaml",
        help="Path to the configuration YAML file. (default: %(default)s)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/demo",
        help="Path to the input data directory. (default: %(default)s)",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="results",
        help="Path to the output directory for saving results. (default: %(default)s)",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="",
        help="The name of the experiment. (default: %(default)s)",
    )
    parser.add_argument(
        "--num_imgs",
        type=int,
        default="-1",
        help="Number of images used. (default: %(default)s)",
    )
    args = parser.parse_args()

    RoboSys = JointReconstructCalib(
        cfg_path=args.cfg_path,
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        exp_name=args.exp_name,
        num_imgs=args.num_imgs,
    )
    RoboSys.run()
