import os, sys
import numpy as np

project_folder = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
)
utils_folder = os.path.join(project_folder, "utils")
sys.path.append(project_folder)
sys.path.append(utils_folder)
from utils.solver import AXXBSolver
from utils.math_op import inv


def main():
    cfg_dict = {"ransac": {"use_ransac": False, "min_samples": 6, "inlier_ratio": 0.5}}
    data_dir = "./data/sim/10_imgs"

    for file in sorted(os.listdir(data_dir)):
        data = np.load(os.path.join(data_dir, file), allow_pickle=True)

        eye2world = data["eye2world"]
        hand2base = data["hand2base"]
        gt_eye2hand = data["gt_eye2hand"]

        solver = AXXBSolver(
            cfg=cfg_dict,
            eye2world_poses=eye2world,
            hand2base_poses=hand2base,
        )
        R_e2h, t_e2h, scale = solver.run()
        As, Bs = solver.As, solver.Bs

        T_e2h = np.eye(4)
        T_e2h[:3, :3] = R_e2h
        T_e2h[:3, 3] = t_e2h

        for A in As:
            A[:3, 3] *= scale

        save_dir = "./results/sim/10_imgs"
        os.makedirs(save_dir, exist_ok=True)
        np.savez(
            os.path.join(save_dir, file[:-4]),
            As=As,
            Bs=Bs,
            gt_h2e=inv(gt_eye2hand),
            ours_h2e=inv(T_e2h),
        )


if __name__ == "__main__":
    main()
