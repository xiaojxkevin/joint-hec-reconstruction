import os
import numpy as np
import pycolmap
import json


def pycolmap_run_mapper(colmap_db_path, recon_path, image_root_path):
    print("<>" * 20)
    print("Running COLMAP mapper")
    # For more details on options, see:
    # https://colmap.github.io/pycolmap/pycolmap.html#pycolmap.IncrementalPipelineOptions
    reconstruction = pycolmap.incremental_mapping(
        database_path=colmap_db_path,
        image_path=image_root_path,
        output_path=recon_path,
        options=pycolmap.IncrementalPipelineOptions({
            "multiple_models": False,
            "extract_colors": True,
            "ba_local_max_refinements": 5,
            "ba_global_max_refinements": 10
        })
    )


def extract_and_save_correspondences(reconstruction_path: str, output_file: str):
    """
    Extract 2D-3D correspondences from a COLMAP reconstruction and save them to a file.
    
    Args:
        reconstruction_path (str): Path to the COLMAP reconstruction folder.
        output_file (str): File path to save the correspondences.
    
    Returns:
        dict: A dictionary containing the correspondence information.
    """
    
    # Load the reconstruction data
    reconstruction = pycolmap.Reconstruction(reconstruction_path)
    if not reconstruction:
        raise RuntimeError(f"Failed to load reconstruction from {reconstruction_path}")
    print("<>" * 20)
    print(reconstruction.summary())
    
    # Dictionary to store all correspondences
    correspondences = {
        "images": {},
        "points3D": {}
    }
    
    # Extract camera parameters
    # In most cases use just use a single camera, thus just ignore the for loop
    cameras = {}
    for camera_id, camera in reconstruction.cameras.items():
        K = np.eye(3)
        K[0, 0] = camera.focal_length_x
        K[1, 1] = camera.focal_length_y
        K[0, 2] = camera.principal_point_x
        K[1, 2] = camera.principal_point_y
        cameras[camera_id] = {
            "intrinsics": K.tolist(),
        }
    correspondences["cameras"] = cameras
    
    # Extract image data and 2D-3D tracks
    for image_id, image in reconstruction.images.items():
        image_name = image.name
        camera_id = image.camera_id
        
        # Get the camera pose (camera-to-world transformation)
        cam_from_world = image.cam_from_world.matrix
        if callable(cam_from_world):
            cam_from_world = cam_from_world()
        
        # Extract 2D points and their corresponding 3D points
        points2D_data = []
        for point2D_idx, point2D in enumerate(image.points2D):
            point_data = {
                "x": float(point2D.xy[0]),
                "y": float(point2D.xy[1]),
                "point3D_id": int(point2D.point3D_id) if point2D.has_point3D() else -1
            }
            points2D_data.append(point_data)
            
            # For reverse lookup, add the observation to the 3D point track
            if point2D.has_point3D():
                point3D_id = int(point2D.point3D_id)
                if point3D_id not in correspondences["points3D"]:
                    point3D = reconstruction.points3D[point3D_id]
                    correspondences["points3D"][point3D_id] = {
                        "id": point3D_id,
                        "xyz": point3D.xyz.tolist(),
                        "color": point3D.color.tolist(),
                        "error": float(point3D.error),
                        "track": []  # List of (image_id, point2D index) pairs
                    }
                # Append this observation to the track
                correspondences["points3D"][point3D_id]["track"].append({
                    "image_id": image_id,
                    "image_name": image_name,
                    "point2D_idx": point2D_idx
                })
        
        # Save image data and associated observations
        correspondences["images"][image_id] = {
            "id": image_id,
            "name": image_name,
            "camera_id": camera_id,
            "cam_from_world": cam_from_world.tolist(),
            "points2D": points2D_data
        }
    
    # Save the correspondences dictionary to a file
    with open(output_file, "w") as f:
        json.dump(correspondences, f, indent=2)
    
    print(f"Saved correspondences to {output_file}")
    print(f"Total images: ", len(correspondences["images"]))
    print(f"Total 3D points: ", len(correspondences["points3D"]))
    
    return correspondences


def process_colmap_data(correspondences):
    # Extract the primary camera intrinsics
    K_lists = []
    for img_info in correspondences["images"].values():
        cam_id = img_info["camera_id"]
        # Ensure that the camera_id is correctly referenced as an integer key
        K_lists.append(correspondences["cameras"][cam_id]["intrinsics"])
    K = np.mean(K_lists, axis=0).reshape(3, 3)  # Average intrinsics to get the main camera matrix
    print("Primary camera intrinsic matrix:\n", K)

    # Save the world-to-camera transformation matrices (n, 4, 4)
    w2c = []
    for i in range(1, len(correspondences["images"]) + 1):
        img_info = correspondences["images"][i]
        mat = np.eye(4)
        mat[:3, :] = np.array(img_info["cam_from_world"]).reshape(3, 4)
        w2c.append(mat)
    w2c = np.asarray(w2c, dtype=np.float64)
    # np.save("world2cam.npy", w2c)
    print(f"Saved camera pose matrices with shape {w2c.shape}")

    # Save the point cloud data (m, 6) and verify the coordinate system
    points = []
    for p3d in correspondences["points3D"].values():
        xyz = np.array(p3d["xyz"])
        rgb = np.array(p3d["color"]) / 255.0  # Normalize color values to [0, 1]
        points.append(np.concatenate([xyz, rgb]))
    points_npy = np.vstack(points)
    np.save("point_cloud.npy", points_npy)
    print(f"Saved point cloud data with shape {points_npy.shape}")
    print("Point cloud coordinates are represented in the world coordinate system")

    return K, w2c, points_npy

if __name__ == "__main__":
    # Example usage
    reconstruction_folder = "results/0318/colmap/reconstruction/0"
    output_file = "./colmap_correspondences.json"
    
    # Extract and save correspondences
    correspondences = extract_and_save_correspondences(reconstruction_folder, output_file)
    
    # Process the COLMAP data
    K, w2c, points_npy = process_colmap_data(correspondences)
    
    print("Intrinsic matrix K:\n", K)
    print("World-to-camera matrices shape:", w2c.shape)
    print("Point cloud data shape:", points_npy.shape)