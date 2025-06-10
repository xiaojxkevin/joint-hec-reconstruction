import os
import numpy as np
import pycolmap
import json
import kapture
from kapture.utils.paths import path_secure
from typing import Union, Tuple, List, Optional
import os.path as path
import PIL
import PIL.Image


def kapture_import_image_folder_or_list(
    images_path: Union[str, Tuple[str, List[str]]],
    camera_matrix: Optional[np.ndarray] = None,
) -> kapture.Kapture:
    """Modified to use known camera intrinsics."""

    images = kapture.RecordsCamera()
    if isinstance(images_path, str):
        images_root = images_path
        file_list = [
            path.relpath(path.join(dirpath, filename), images_root)
            for dirpath, dirs, filenames in os.walk(images_root)
            for filename in filenames
        ]
        file_list = sorted(file_list)
    else:
        images_root, file_list = images_path

    sensors = kapture.Sensors()
    try:
        with PIL.Image.open(path.join(images_root, file_list[0])) as im:
            width, height = im.size
    except (OSError, PIL.UnidentifiedImageError):
        raise RuntimeError(f"Invalid image file {file_list[0]}")

    if camera_matrix is not None:
        assert camera_matrix.shape == (
            3,
            3,
        ), f"Camera matrix shape {camera_matrix.shape}"
        fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
        cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
        model_params = [width, height, fx, fy, cx, cy]
        camera_type = kapture.CameraType.PINHOLE
    else:
        # TODO check the type of the camera
        raise NotImplementedError("Camera intrinsics are not provided.")
        model_params = [width, height]
        camera_type = kapture.CameraType.UNKNOWN_CAMERA

    camera_id = "sensor"
    if camera_id not in sensors:
        sensors[camera_id] = kapture.Camera(camera_type, model_params)

    for n, filename in enumerate(file_list):
        images[(n, camera_id)] = path_secure(filename)

    return kapture.Kapture(sensors=sensors, records_camera=images)


def pycolmap_run_mapper(
    colmap_db_path: str, recon_path: str, image_root_path: str, opt_K: bool
):
    """when opt_K is true, we optimize the camera intrinsics during mapping"""
    print("<>" * 20)
    print("Running COLMAP mapper")
    # For more details on options, see:
    # https://colmap.github.io/pycolmap/pycolmap.html#pycolmap.IncrementalPipelineOptions
    reconstruction = pycolmap.incremental_mapping(
        database_path=colmap_db_path,
        image_path=image_root_path,
        output_path=recon_path,
        options=pycolmap.IncrementalPipelineOptions(
            {
                "multiple_models": False,
                "ba_refine_focal_length": opt_K,
                "ba_refine_extra_params": opt_K,
                "extract_colors": True,
                "ba_local_max_refinements": 5,
                "ba_global_max_refinements": 10,
            }
        ),
    )


def extract_and_save_correspondences(reconstruction_path: str, output_file: str | None):
    """
    Extract 2D-3D correspondences from a COLMAP reconstruction and save them to a file.

    Args:
        reconstruction_path (str): Path to the COLMAP reconstruction folder.
        output_file (str): File path to save the correspondences.

    Returns:
        dict: A dictionary containing the correspondence information.
    """

    # Load the reconstruction data
    # https://colmap.github.io/pycolmap/pycolmap.html#pycolmap.Reconstruction
    reconstruction = pycolmap.Reconstruction(reconstruction_path)
    if not reconstruction:
        raise RuntimeError(f"Failed to load reconstruction from {reconstruction_path}")
    print("<>" * 20)
    print(reconstruction.summary())

    # Dictionary to store all correspondences
    correspondences = {"cameras": {}, "images": {}, "points3D": {}}

    # Extract camera parameters
    # In most cases we just use a single camera, thus just ignore the for loop
    for camera_id, camera in reconstruction.cameras.items():
        print("*" * 50, "\n", camera, "\n", "*" * 50)
        K = np.eye(3)
        K[0, 0] = camera.focal_length_x
        K[1, 1] = camera.focal_length_y
        K[0, 2] = camera.principal_point_x
        K[1, 2] = camera.principal_point_y
        correspondences["cameras"] = {
            "intrinsics": K.tolist(),
        }
        break

    # 3D points
    pts3d_idx = 0
    for i, (point3D_id, point3D) in enumerate(reconstruction.points3D.items()):
        if float(point3D.error) > 2.0:
            continue
        correspondences["points3D"][int(point3D_id)] = {
            "sorted_id": pts3d_idx,
            "xyz": point3D.xyz.tolist(),
            "color": point3D.color.tolist(),
            "error": float(point3D.error),
        }
        pts3d_idx += 1

    # Extract 2D-3D correspondences
    points2D = []
    pts2d_idx = 0
    for image_id, image in reconstruction.images.items():
        # if not image.has_pose():
        #     continue
        # Get the camera pose (camera-to-world transformation)
        cam_from_world = image.cam_from_world.matrix
        if callable(cam_from_world):
            cam_from_world = cam_from_world()

        # Extract 2D points and their corresponding 3D points
        pts2d_indices, pts3d_indices = [], []
        for point2D_idx, point2D in enumerate(image.points2D):
            if not point2D.has_point3D():
                continue
            pt_id = int(point2D.point3D_id)
            if pt_id in correspondences["points3D"]:
                points2D.append((float(point2D.xy[0]), float(point2D.xy[1])))
                pts2d_indices.append(pts2d_idx)
                pts3d_indices.append(correspondences["points3D"][pt_id]["sorted_id"])
                pts2d_idx += 1
        assert len(pts2d_indices) == len(
            pts3d_indices
        ), f"Length mismatch: {len(pts2d_indices)} != {len(pts3d_indices)}"
        correspondences["images"][int(image_id) - 1] = {
            "wolrd_to_cam": cam_from_world.tolist(),
            "pts2d_indices": pts2d_indices,
            "pts3d_indices": pts3d_indices,
        }

    # 2D points
    correspondences["points2D"] = points2D

    # Save the correspondences dictionary to a file
    if output_file is not None:
        with open(output_file, "w") as f:
            json.dump(correspondences, f, indent=2)

    print(f"Saved correspondences to {output_file}")
    print(f"Total images: ", len(correspondences["images"]))
    print(f"Total 2D points: ", len(correspondences["points2D"]))
    print(f"Total 3D points: ", len(correspondences["points3D"]))

    return correspondences


def process_colmap_data(correspondences):
    """
    Returns the camera intrinsic matrix, world-to-camera transformation matrices, 3D points, 2D points, and visibility.
    """
    if isinstance(correspondences, str):
        # Load from file if a string path is provided
        with open(correspondences, "r") as f:
            correspondences = json.load(
                f,
                object_hook=lambda d: {
                    int(k) if k.isdigit() else k: v for k, v in d.items()
                },
            )
    # Extract the primary camera intrinsics
    K = np.array(correspondences["cameras"]["intrinsics"]).reshape(3, 3)
    print("Primary camera intrinsic matrix:\n", K)

    # Save 2D points
    points2D = np.asarray(correspondences["points2D"])
    print(f"Saved 2d points with shape {points2D.shape}")

    # Save 3D points
    points3D_dict = correspondences["points3D"]
    points3D = np.zeros((len(points3D_dict), 7), dtype=np.float32)
    for point_id in points3D_dict.keys():
        point = points3D_dict[point_id]
        sorted_id = point["sorted_id"]
        points3D[sorted_id, 0:3] = point["xyz"]
        points3D[sorted_id, 3:6] = np.asarray(point["color"]) / 255.0
        points3D[sorted_id, 6] = point["error"]
    # np.save("tmp_point_cloud.npy", points3D)
    print(f"Saved point cloud data with shape {points3D.shape}")
    print("Point cloud coordinates are represented in the world coordinate system")

    # Save 3D-2D correspondences
    images_dict = correspondences["images"]
    world2cam_poses = []
    visibility = {}
    for image_id in sorted(images_dict.keys()):
        img_info = correspondences["images"][image_id]

        # Save eye poses
        mat = np.eye(4)
        mat[:3, :] = np.array(img_info["wolrd_to_cam"]).reshape(3, 4)
        world2cam_poses.append(mat)

        visibility[image_id] = {
            "pts2d_indices": img_info["pts2d_indices"],
            "pts3d_indices": img_info["pts3d_indices"],
        }

    return K, np.asarray(world2cam_poses), points3D, points2D, visibility


if __name__ == "__main__":
    # Example usage
    reconstruction_folder = "results/no_chessboard/000_10/colmap/reconstruction/0"
    # output_file = "./colmap_correspondences.json"
    output_file = None

    # Extract and save correspondences
    correspondences = extract_and_save_correspondences(
        reconstruction_folder, output_file
    )

    # Process the COLMAP data
    K, w2c, points3D, points2D, visibility = process_colmap_data(correspondences)
    print(visibility.keys())

    print("Intrinsic matrix K:\n", K)
    print("World-to-camera matrices shape:", w2c.shape)
    print("Point cloud data shape:", points3D.shape)
    print("2D points shape:", points2D.shape)
