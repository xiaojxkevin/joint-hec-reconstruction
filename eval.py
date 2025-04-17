import numpy as np
import json
import os
import matplotlib.pyplot as plt
from copy import deepcopy
import seaborn as sns
import pandas as pd


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
    motion_dict = {
        "08": {"t": [], "R": []},
        "10": {"t": [], "R": []},
        "12": {"t": [], "R": []},
    }
    noise_dict = {
        "0.00": {"t": [], "R": []},
        "0.50": {"t": [], "R": []},
        "1.00": {"t": [], "R": []},
        "1.50": {"t": [], "R": []},
    }
    errors = {
        "init": {"motion": motion_dict, "noise": noise_dict},
        "ba": {"motion": deepcopy(motion_dict), "noise": deepcopy(noise_dict)},
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


def plot(errors, cond):
    """
    Display side-by-side violin plots of translation and rotation errors
    for both init vs. BA phases under the given condition.

    Parameters:
      errors:     dict storing error data, e.g. errors["init"]["motion"]["category"]["t"]
      cond:       str, either "motion" or "noise"
    """
    # Build a long‑form DataFrame with columns: Category, Phase, Error, ErrorType
    records = []
    for phase in ("init", "ba"):
        phase_label = "Initial" if phase == "init" else "BA"
        for cat, data_dict in errors[phase][cond].items():
            for etype, et_label in (("t", "Translation"), ("R", "Rotation")):
                vals = data_dict.get(etype, [])
                for v in vals:
                    records.append(
                        {
                            "Category": cat,
                            "Phase": phase_label,
                            "Error": v,
                            "ErrorType": et_label,
                        }
                    )

    df = pd.DataFrame.from_records(records)
    if df.empty:
        raise ValueError(f"No error data found for condition '{cond}'")

    # Determine x‑axis label based on condition
    xlabel = "Number of motions" if cond == "motion" else "Gaussian noise level (sigma)"

    # Prepare figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=False)

    # Plot settings for each error type
    for ax, (etype, ylabel) in zip(
        axes,
        [
            ("Translation", "Translation Error (m)"),
            ("Rotation", "Rotation Error (degrees)"),
        ],
    ):
        subset = df[df["ErrorType"] == etype]
        sns.violinplot(
            x="Category",
            y="Error",
            hue="Phase",
            data=subset,
            split=True,
            inner="quartile",
            palette={"Initial": "skyblue", "BA": "salmon"},
            cut=0,
            ax=ax,
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{etype} Errors ({cond.capitalize()} Condition)")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        # only show legend on the first subplot
        ax.legend(title="Phase")

    plt.tight_layout()
    plt.savefig(f"{cond}.png")


def main():
    errors = compute_errors("no_chessboard", 50)
    plot(errors, cond="motion")
    plot(errors, cond="noise")


if __name__ == "__main__":
    main()
