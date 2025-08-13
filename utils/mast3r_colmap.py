import sys, os
import os.path as path
import numpy as np
from scipy.spatial.transform import Rotation as R
import re
import shutil
import pycolmap
import kapture
from kapture.converter.colmap.database_extra import kapture_to_colmap
from kapture.converter.colmap.database import COLMAPDatabase
from kapture.utils.paths import path_secure
from typing import Union, Tuple, List, Optional
import PIL
import PIL.Image
import json
import logging


# Determine the project folder and add necessary paths to sys.path
project_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
mast3r_local_dir = os.path.join(project_folder, "mast3r")
sys.path.append(mast3r_local_dir)
sys.path.append(project_folder)

from mast3r.colmap.mapping import run_mast3r_matching
from mast3r.image_pairs import make_pairs
from mast3r.model import AsymmetricMASt3R
from dust3r.utils.image import load_images
from utils.math_op import inv


def get_img_lists(img_dir: str, used_indices: np.ndarray) -> list:
    # Return a sorted list of image paths (png, jpg, jpeg) limited to num_imgs
    return np.asarray(
        [
            os.path.join(img_dir, f)
            for f in sorted(os.listdir(img_dir))
            if re.match(r".*\.(png|jpg|jpeg)$", f)
        ]
    )[used_indices].tolist()


class MAST3R_COLMAP:
    def __init__(self, cfg: dict) -> None:
        self.input_dir: str = cfg["input_dir"]
        self.output_dir: str = cfg["output_dir"]
        self.model_path: str = cfg["model_path"]
        self.used_indices: np.ndarray = cfg["used_indices"]
        self.intrinsics: np.ndarray = cfg.get("intrinsics", None)
        self.image_size: int = cfg.get("image_size", 512)
        self.device: str = cfg.get("device", "cuda")

        self.logger = logging.getLogger("hand_eye_calibration")

    def run(self):
        recon_path = os.path.join(self.output_dir, "reconstruction")
        reconstruction_folder = os.path.join(recon_path, "0")
        correspondence_file = os.path.join(self.output_dir, "colmap_raw.json")
        if os.path.exists(correspondence_file):
            self.logger.info("Loading existing correspondences")
            correspondence_data = self.extract_and_save_correspondences(
                reconstruction_folder, correspondence_file
            )
            K, world2cam, points3D, points2D, visibility = self.process_correspondences(
                correspondence_data
            )
            cam2world = np.asarray([inv(p) for p in world2cam])
            return K, cam2world, points3D, points2D, visibility

        # Create output directory if it does not exist
        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize the MASt3R model with pre-trained weights
        model = AsymmetricMASt3R.from_pretrained(self.model_path).to(self.device)

        # Load images and generate a kapture data structure
        img_path_lists = get_img_lists(self.input_dir, self.used_indices)
        img_relpath = [
            os.path.relpath(filename, self.input_dir) for filename in img_path_lists
        ]
        imgs = load_images(img_path_lists, size=self.image_size)

        # Prepare image pairs for matching
        pairs = make_pairs(imgs, scene_graph="complete", symmetrize=True)
        kdata = self.kapture_import_image_folder_or_list(
            (self.input_dir, img_relpath), camera_matrix=self.intrinsics
        )
        image_names = kdata.records_camera.data_list()
        image_pairs = [
            (img_relpath[img1["idx"]], img_relpath[img2["idx"]]) for img1, img2 in pairs
        ]

        ############################################ Create COLMAP database
        db_path = os.path.join(self.output_dir, "colmap.db")
        if os.path.exists(db_path):
            os.remove(db_path)
        colmap_db = COLMAPDatabase.connect(db_path)

        try:
            # Export kapture data to the COLMAP database
            kapture_to_colmap(
                kapture_data=kdata,
                kapture_dirpath=self.input_dir,
                tar_handler=None,
                database=colmap_db,
            )
            # Run MASt3R matching to generate image pairs for COLMAP
            colmap_image_pairs = run_mast3r_matching(
                model=model,
                maxdim=self.image_size,
                patch_size=16,
                device=self.device,
                kdata=kdata,
                root_path=self.input_dir,
                image_pairs_kapture=image_pairs,
                colmap_db=colmap_db,
                dense_matching=False,
                pixel_tol=5,
                conf_thr=1.001,
                skip_geometric_verification=False,
                min_len_track=3,
            )
            colmap_db.close()
        except Exception as e:
            print(f"Error during reconstruction: {str(e)}")
            colmap_db.close()
            raise e
        if len(colmap_image_pairs) == 0:
            raise Exception("no matches were kept")

        ############################################ Verify matches and run reconstruction
        with open(os.path.join(self.output_dir, "pairs.txt"), "w") as f:
            for image_path1, image_path2 in colmap_image_pairs:
                f.write("{} {}\n".format(image_path1, image_path2))
        pycolmap.verify_matches(db_path, os.path.join(self.output_dir, "pairs.txt"))
        if os.path.isdir(recon_path):
            shutil.rmtree(recon_path)
        os.makedirs(recon_path, exist_ok=True)
        opt_K = True if self.intrinsics is None else False
        self.pycolmap_run_mapper(db_path, recon_path, self.input_dir, opt_K)

        ############################################ Save results
        # Load the reconstruction results from COLMAP
        # Extract and save correspondences between 2D and 3D points
        correspondence_data = self.extract_and_save_correspondences(
            reconstruction_folder, correspondence_file
        )
        K, world2cam, points3D, points2D, visibility = self.process_correspondences(
            correspondence_data
        )
        cam2world = np.asarray([inv(p) for p in world2cam])

        return K, cam2world, points3D, points2D, visibility

    def kapture_import_image_folder_or_list(
        self,
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
        self, colmap_db_path: str, recon_path: str, image_root_path: str, opt_K: bool
    ):
        """when opt_K is true, we optimize the camera intrinsics during mapping"""
        self.logger.info("Running COLMAP mapper")
        # Show only warnings
        pycolmap.logging.minloglevel = pycolmap.logging.Level.WARNING
        # For more details on options, see:
        # https://colmap.github.io/pycolmap/pycolmap.html#pycolmap.IncrementalPipelineOptions
        reconstruction = pycolmap.incremental_mapping(
            database_path=colmap_db_path,
            image_path=image_root_path,
            output_path=recon_path,
            options=pycolmap.IncrementalPipelineOptions(
                {
                    "num_threads": 8,
                    "multiple_models": False,
                    "ba_refine_focal_length": opt_K,
                    "ba_refine_extra_params": opt_K,
                    "extract_colors": True,
                    "ba_local_max_refinements": 5,
                    "ba_global_max_refinements": 10,
                    "mapper": {
                        "num_threads": 8,
                    },
                }
            ),
        )

    def extract_and_save_correspondences(
        self, reconstruction_path: str, output_file: str | None
    ):
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
            raise RuntimeError(
                f"Failed to load reconstruction from {reconstruction_path}"
            )
        self.logger.info("Reconstruction summary:\n%s", reconstruction.summary())

        # Dictionary to store all correspondences
        correspondences = {"cameras": {}, "images": {}, "points3D": {}}

        # Extract camera parameters
        # In most cases we just use a single camera, thus just ignore the for loop
        for camera_id, camera in reconstruction.cameras.items():
            self.logger.info("camera details:\n%s", camera)
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
        # https://colmap.github.io/pycolmap/pycolmap.html#pycolmap.Point3D
        pts3d_idx = 0
        for i, (point3D_id, point3D) in enumerate(reconstruction.points3D.items()):
            # Add a few constraints
            if float(point3D.error) > 1.0 and point3D.track.length() <= 2:
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
            # TODO detail with the case where an image is invalid
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
                    pts3d_indices.append(
                        correspondences["points3D"][pt_id]["sorted_id"]
                    )
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

        self.logger.info("Saved correspondences to %s", output_file)
        self.logger.info("Total images: %d", len(correspondences["images"]))
        self.logger.info("Total 2D points: %d", len(correspondences["points2D"]))
        self.logger.info("Total 3D points: %d", len(correspondences["points3D"]))

        return correspondences

    def process_correspondences(self, correspondences: dict):
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
        self.logger.info("Primary camera intrinsic matrix:\n%s", K)

        # Save 2D points
        points2D = np.asarray(correspondences["points2D"])
        self.logger.info("Saved 2d points with shape:\n%s", points2D.shape)

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
        self.logger.info(
            "Point cloud coordinates are represented in the world coordinate system and saved with shape\n%s",
            points3D.shape,
        )

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
    pass
