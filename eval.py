import numpy as np
import json
import os
import matplotlib.pyplot as plt


def compute_translation_error(T_gt, T_est):
    """
    Computes the Euclidean distance between the translation components of two poses.
    """
    t_gt = T_gt[0:3, 3]
    t_est = T_est[0:3, 3]
    error = np.linalg.norm(t_gt - t_est)
    return error


def compute_rotation_error(T_gt, T_est):
    """
    Computes the angular difference between the rotation parts of two poses.
    """
    R_gt = T_gt[0:3, 0:3]
    R_est = T_est[0:3, 0:3]
    # Relative rotation matrix
    R_diff = R_gt.T @ R_est
    from scipy.spatial.transform import Rotation as R

    vec = R.from_matrix(R_diff).as_rotvec()
    angle = np.linalg.norm(vec)
    return np.degrees(angle)


def compute_error(T_gt, T_est):
    """
    Computes the translation and rotation errors between two poses.
    """
    t_error = compute_translation_error(T_gt, T_est)
    R_error = compute_rotation_error(T_gt, T_est)
    return t_error, R_error


def compute_errors(folder, num_tests):
    motion_dict = (
        {
            "08": {"t": [], "R": []},
            "10": {"t": [], "R": []},
            "12": {"t": [], "R": []},
        },
    )
    noise_dict = {
        "0.00": {"t": [], "R": []},
        "0.50": {"t": [], "R": []},
        "1.00": {"t": [], "R": []},
        "1.50": {"t": [], "R": []},
    }
    errors = {
        "init": {"motion": motion_dict, "noise": noise_dict},
        "ba": {"motion": motion_dict, "noise": noise_dict},
    }
    for i in range(num_tests):
        test_id = f"{i:03d}"
        gt_path = os.path.join(
            f"data/{folder}",
            test_id,
            "motion",
            "eye2hand_pose.txt",
        )
        for phase in errors.keys():
            for cond in errors[phase].keys():
                for subcond in errors[phase][cond].keys():
                    if cond == "noise":
                        suffix = f"{subcond}_12"
                    else:
                        suffix = subcond

                    pre_path = os.path.join(
                        f"results/{folder}",
                        test_id,
                        f"{cond}_{suffix}",
                        f"{phase}_T_eye2hand.txt",
                    )
                    gt_pose = np.loadtxt(gt_path)
                    pre_pose = np.loadtxt(pre_path)
                    t_error, R_error = compute_error(gt_pose, pre_pose)
                    errors[phase][cond][subcond]["t"].append(t_error)
                    errors[phase][cond][subcond]["R"].append(R_error)
    return errors


def plot_error_bars_two_phases(errors, cond, error_type="t"):
    """
    Display error bar charts for both the init and ba phases under a specified condition in the same figure.

    Parameters:
      errors: Dictionary storing all error data
      cond: Specified condition, either "motion" or "noise"
      error_type: Type of error to display, "t" for translation error, "R" for rotation error
    """
    # Get sub-conditions, e.g., for motion: ["08", "10", "12"], for noise: ["0.00", "0.50", "1.00"]
    categories = list(errors["init"][cond].keys())

    means_init, stds_init = [], []
    means_ba, stds_ba = [], []

    for cat in categories:
        # Data for the init phase
        data_init = errors["init"][cond][cat][error_type]
        if data_init:
            means_init.append(np.mean(data_init))
            stds_init.append(np.std(data_init))
        else:
            means_init.append(0)
            stds_init.append(0)

        # Data for the ba phase
        data_ba = errors["ba"][cond][cat][error_type]
        if data_ba:
            means_ba.append(np.mean(data_ba))
            stds_ba.append(np.std(data_ba))
        else:
            means_ba.append(0)
            stds_ba.append(0)

    # Set bar chart parameters
    x = np.arange(len(categories))
    bar_width = 0.35

    plt.figure(figsize=(10, 6))

    # Plot error bars for init and ba phases separately
    plt.bar(
        x - bar_width / 2,
        means_init,
        bar_width,
        yerr=stds_init,
        capsize=5,
        alpha=0.7,
        label="init",
        color="skyblue",
        edgecolor="black",
    )
    plt.bar(
        x + bar_width / 2,
        means_ba,
        bar_width,
        yerr=stds_ba,
        capsize=5,
        alpha=0.7,
        label="ba",
        color="salmon",
        edgecolor="black",
    )

    # Set x-axis labels, title, and legend
    plt.xticks(x, categories)
    ylabel = (
        "Translation Error (m)" if error_type == "t" else "Rotation Error (degrees)"
    )
    plt.xlabel(f"{cond} condition")
    plt.ylabel(ylabel)
    plt.title(f"Error in init and ba phases under {cond} condition")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.legend()
    plt.savefig(f"{cond}_{error_type}.png")


def main():
    errors = compute_errors("no_chessboard", 20)
    plot_error_bars_two_phases(errors, cond="motion", error_type="t")
    plot_error_bars_two_phases(errors, cond="motion", error_type="R")
    plot_error_bars_two_phases(errors, cond="noise", error_type="t")
    plot_error_bars_two_phases(errors, cond="noise", error_type="R")


if __name__ == "__main__":
    main()
