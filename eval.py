import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.spatial.transform import Rotation as R

# ——— USER: point this at your ground‐truth directory ———
GT_ROOT = "data/no_chessboard"  # e.g. mirrors results/no_chessboard structure
gt_filename = "eye2hand_pose.txt"  # or "gt_T_eye2hand.txt", etc.
# ——————————————————————————————————————————————


def load_pose(path):
    return np.loadtxt(path)


def compute_translation_error(T_gt, T_est):
    return np.linalg.norm(T_gt[:3, 3] - T_est[:3, 3])


def compute_rotation_error(T_gt, T_est):
    R_diff = T_gt[:3, :3].T @ T_est[:3, :3]
    angle = np.linalg.norm(R.from_matrix(R_diff).as_rotvec())
    return np.degrees(angle)


def compute_all_errors():
    """
    Walks results/no_chessboard/…/<test_id>/<img_folder>/init_T… and ba_T…
    and compares to GT_ROOT/…/<test_id>/<img_folder>/{gt_filename}
    """
    records = []
    base = "results/no_chessboard"
    for combo in sorted(os.listdir(base)):  # e.g. indoor-marble-3_objs
        combo_dir = os.path.join(base, combo)
        print("reading")
        if not os.path.isdir(combo_dir):
            continue

        for test_id in sorted(os.listdir(combo_dir)):  # 01-50
            test_dir = os.path.join(combo_dir, test_id)
            if not os.path.isdir(test_dir):
                continue

            for img_folder in sorted(os.listdir(test_dir)):  # e.g. 06_imgs
                if not img_folder.endswith("_imgs"):
                    continue
                img_dir = os.path.join(test_dir, img_folder)

                # paths
                gt_path = os.path.join(GT_ROOT, combo, test_id, gt_filename)
                init_path = os.path.join(img_dir, "init_T_eye2hand.txt")
                ba_path = os.path.join(img_dir, "ba_T_eye2hand.txt")

                if not (
                    os.path.exists(gt_path)
                    and os.path.exists(init_path)
                    and os.path.exists(ba_path)
                ):
                    print(f"Skipping missing: {combo}/{test_id}/{img_folder}")
                    continue

                T_gt = load_pose(gt_path)
                T_init = load_pose(init_path)
                T_ba = load_pose(ba_path)

                t_init = compute_translation_error(T_gt, T_init)
                r_init = compute_rotation_error(T_gt, T_init)
                t_ba = compute_translation_error(T_gt, T_ba)
                r_ba = compute_rotation_error(T_gt, T_ba)

                # improvement: (init - ba) / init
                imp_t = (t_init - t_ba) / t_init if t_init != 0 else np.nan
                imp_r = (r_init - r_ba) / r_init if r_init != 0 else np.nan

                records.append(
                    {
                        "combo": combo,
                        "test_id": test_id,
                        "images": int(img_folder.split("_")[0]),  # 6,7,8,9,10
                        "phase": "init",
                        "t_error": t_init,
                        "r_error": r_init,
                        "imp_t": imp_t,
                        "imp_r": imp_r,
                    }
                )
                records.append(
                    {
                        "combo": combo,
                        "test_id": test_id,
                        "images": int(img_folder.split("_")[0]),
                        "phase": "ba",
                        "t_error": t_ba,
                        "r_error": r_ba,
                        "imp_t": imp_t,
                        "imp_r": imp_r,
                    }
                )

    return pd.DataFrame(records)


def plot_errors(df):
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 7), sharex=True)

    for ax, (err_type, ylabel) in zip(
        axes, [("t_error", "Translation Error (m)"), ("r_error", "Rotation Error (°)")]
    ):
        sns.boxplot(
            x="images",
            y=err_type,
            hue="phase",
            data=df,
            ax=ax,
            width=0.7,
            fliersize=2.0,
        )
        ax.set_xlabel("Number of Used Images")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.legend()

    plt.tight_layout()
    plt.savefig(f"error_boxplot.svg")
    plt.close()


def plot_improvements(df):
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid")

    df_imp = df[df.phase == "ba"].copy()
    for imp_col, ylabel in [
        ("imp_t", "Translation Improvement"),
        ("imp_r", "Rotation Improvement"),
    ]:
        plt.figure(figsize=(8, 6))
        ax = sns.boxplot(
            x="images",
            y=imp_col,
            data=df_imp,
            palette="coolwarm",
        )
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Number of Used Images")
        ax.set_ylabel(ylabel + "%")
        ax.set_title(f"Percentage of {ylabel} after BA")
        plt.tight_layout()
        plt.savefig(f"improvement_{imp_col}.svg")
        plt.close()


def main():
    # df = compute_all_errors()
    # if df.empty:
    #     raise RuntimeError("No data found—check your paths!")
    # df.to_csv("./out", index=False)

    df = pd.read_csv("./out")
    plot_errors(df)
    # plot_improvements(df)
    print(
        "Saved: error_t_error.svg, error_r_error.svg, improvement_t.svg, improvement_r.svg"
    )


if __name__ == "__main__":
    main()
