import os, sys
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation


project_folder = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
)
utils_folder = os.path.join(project_folder, "utils")
sys.path.append(project_folder)
sys.path.append(utils_folder)
from utils.metrics import compute_error


def params_to_se3(params):
    """Converts a 6-element vector (3 for axis-angle, 3 for translation)
    into a 4x4 SE(3) matrix."""

    # Extract rotation (axis-angle) and translation
    axis_angle = params[:3]
    translation = params[3:]

    # Create the 3x3 rotation matrix from axis-angle
    # The `from_rotvec` method directly interprets the vector as axis-angle
    rotation_matrix = Rotation.from_rotvec(axis_angle).as_matrix()

    # Create the 4x4 transformation matrix
    T = np.eye(4)
    T[:3, :3] = rotation_matrix
    T[:3, 3] = translation

    return T


def se3_to_params(T):
    """Converts a 4x4 SE(3) matrix into a 6-element vector (3 for axis-angle, 3 for translation)."""

    # Extract the rotation matrix and translation vector
    rotation_matrix = T[:3, :3]
    translation = T[:3, 3]

    # Convert the rotation matrix to axis-angle representation
    axis_angle = Rotation.from_matrix(rotation_matrix).as_rotvec()

    # Combine axis-angle and translation into a single vector
    params = np.concatenate((axis_angle, translation))

    return params


def cost_function(params, A_list, B_list):
    """Calculates the total error for the AX=XB problem."""

    # Convert the current parameter vector to the transformation matrix X
    X = params_to_se3(params)

    total_error = 0.0

    # Loop through all corresponding pairs of A and B matrices
    for A, B in zip(A_list, B_list):
        # Calculate the error matrix for the current pair
        error_matrix = A @ X - X @ B

        # Add the squared Frobenius norm of the error to the total
        # np.linalg.norm(..., 'fro')**2 is equivalent and also works
        total_error += np.sum(error_matrix**2)

    return total_error


def test_one(data: np.ndarray):

    A_matrices = data["As"]
    B_matrices = data["Bs"]
    X_true = data["gt_h2e"]
    X_ours = data["ours_h2e"]
    x0 = se3_to_params(X_ours)

    # result = minimize(
    #     fun=cost_function,
    #     x0=x0,
    #     args=(A_matrices, B_matrices),
    #     method="Nelder-Mead",
    #     options={"maxiter": 1000, "disp": False},
    #     tol=1e-4,
    # )
    result = minimize(
        fun=cost_function,
        x0=x0,
        args=(A_matrices, B_matrices),
        jac="3-point",
        method="SLSQP",
        options={"maxiter": 1000, "disp": False},
        tol=1e-4,
    )

    # --- Retrieve and display the result ---
    if result.success:
        # The optimized parameter vector
        optimized_params = result.x

        # Convert it back to the final SE(3) matrix
        X_optimized = params_to_se3(optimized_params)

    else:
        print("\nOptimization failed.")
        print(f"Message: {result.message}")

    return {
        "X_optimized": X_optimized,
        "X_true": X_true,
        "X_ours": X_ours,
        "success": result.success,
    }


def main():
    # Assume the directory exists and contains the data files
    data_dir = "./results/sim/10_imgs"
    if not os.path.exists(data_dir):
        print(f"Directory not found: {data_dir}. Please create it and add data files.")
        return

    # --- Step 1: Initialize lists to store errors ---
    errors_before_trans = []
    errors_before_rot = []
    errors_after_trans = []
    errors_after_rot = []

    for file in sorted(os.listdir(data_dir)):

        data = np.load(os.path.join(data_dir, file), allow_pickle=True)
        result = test_one(data)

        t_err_before, r_err_before = compute_error(result["X_true"], result["X_ours"])
        t_err_after, r_err_after = compute_error(
            result["X_true"], result["X_optimized"]
        )

        # Append errors to their respective lists
        errors_before_trans.append(t_err_before)
        errors_before_rot.append(r_err_before)
        errors_after_trans.append(t_err_after)
        errors_after_rot.append(r_err_after)

        # Print results for the current file
        # print(f"File: {file}")
        # print(
        #     f"Error before optimization: Translation={t_err_before:.4f} m, Rotation={r_err_before:.4f} deg"
        # )
        # print(
        #     f"Error after optimization:  Translation={t_err_after:.4f} m, Rotation={r_err_after:.4f} deg\n"
        # )

    # --- Step 3: Calculate and print summary statistics ---
    if not errors_before_trans:
        print("No data files were processed. Cannot compute statistics.")
        return

    print("--- Summary Statistics ---\n")

    # Before Optimization
    mean_trans_before = np.mean(errors_before_trans)
    median_trans_before = np.median(errors_before_trans)
    mean_rot_before = np.mean(errors_before_rot)
    median_rot_before = np.median(errors_before_rot)

    # After Optimization
    mean_trans_after = np.mean(errors_after_trans)
    median_trans_after = np.median(errors_after_trans)
    mean_rot_after = np.mean(errors_after_rot)
    median_rot_after = np.median(errors_after_rot)

    print("--- Before Optimization ---")
    print(
        f"Translation Error | Mean: {mean_trans_before:.4f} m, Median: {median_trans_before:.4f} m"
    )
    print(
        f"Rotation Error    | Mean: {mean_rot_before:.4f} deg, Median: {median_rot_before:.4f} deg\n"
    )

    print("--- After Optimization ---")
    print(
        f"Translation Error | Mean: {mean_trans_after:.4f} m, Median: {median_trans_after:.4f} m"
    )
    print(
        f"Rotation Error    | Mean: {mean_rot_after:.4f} deg, Median: {median_rot_after:.4f} deg"
    )


if __name__ == "__main__":
    main()
