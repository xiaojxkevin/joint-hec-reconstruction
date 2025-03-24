import sys, os
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
import re
import shutil
import pycolmap
import kapture
from kapture.converter.colmap.database_extra import kapture_to_colmap
from kapture.converter.colmap.database import COLMAPDatabase

# Determine the project folder and add necessary paths to sys.path
project_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
mast3r_local_dir = os.path.join(project_folder, "mast3r")
sys.path.append(mast3r_local_dir)
sys.path.append(project_folder)

from mast3r.colmap.mapping import (
    kapture_import_image_folder_or_list,
    run_mast3r_matching
)
from mast3r.image_pairs import make_pairs
from mast3r.model import AsymmetricMASt3R
from dust3r.utils.image import load_images
from utils.colmap_utils import (
    pycolmap_run_mapper,
    extract_and_save_correspondences,
    process_colmap_data
)
from utils.math_op import inv

class Scene:
    def __init__(self, world_to_cam, intrinsics, points3d):
        self.world_to_cam = world_to_cam
        self.intrinsics = intrinsics
        self.points3d = points3d

def get_img_lists(img_dir: str, num_imgs: int) -> list:
    # Return a sorted list of image paths (png, jpg, jpeg) limited to num_imgs
    return [
        os.path.join(img_dir, f)
        for f in sorted(os.listdir(img_dir))
        if re.match(r".*\.(png|jpg|jpeg)$", f)
    ][:num_imgs]

def run_mast3r(input_dir: str,
               output_dir: str,
               model_path: str,
               num_imgs: int = 8,
               image_size: int = 512,
               device: str = "cuda"):
    """
    Runs the MASt3R pipeline to return camera intrinsics, cam2world transformations, and 3D points.
    
    Args:
        input_dir (str): Directory containing input images.
        output_dir (str): Directory to save output files.
        model_path (str): Path to the pre-trained MASt3R model.
        num_imgs (int, optional): Number of images to process. Defaults to 8.
        image_size (int, optional): Size to which images will be resized. Defaults to 512.
        device (str, optional): Device to run the model on ("cuda" or "cpu"). Defaults to "cuda".
    
    Returns:
        tuple: Camera intrinsics, camera-to-world transformation matrices, and 3D points.
    """
    # Create output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    # Initialize the MASt3R model with pre-trained weights
    model = AsymmetricMASt3R.from_pretrained(model_path).to(device)

    # Load images and generate a kapture data structure
    img_path_lists = get_img_lists(input_dir, num_imgs)
    img_relpath = [
        os.path.relpath(filename, input_dir)
        for filename in img_path_lists
    ]
    imgs = load_images(img_path_lists, size=image_size)
    
    # Prepare image pairs for matching
    pairs = make_pairs(imgs, scene_graph="complete", symmetrize=True)
    kdata = kapture_import_image_folder_or_list((input_dir, img_relpath), 
                                                use_single_camera=True)
    image_names = kdata.records_camera.data_list()
    image_pairs = [
        (img_relpath[img1["idx"]], img_relpath[img2["idx"]])
        for img1, img2 in pairs
    ]

    ############################################ Create COLMAP database
    db_path = os.path.join(output_dir, "colmap.db")
    if os.path.exists(db_path):
        os.remove(db_path)
    colmap_db = COLMAPDatabase.connect(db_path)
    
    try:
        # Export kapture data to the COLMAP database
        kapture_to_colmap(kapture_data=kdata, 
                          kapture_dirpath=input_dir, 
                          tar_handler=None, 
                          database=colmap_db)
        # Run MASt3R matching to generate image pairs for COLMAP
        colmap_image_pairs = run_mast3r_matching(
            model=model,
            maxdim=image_size,
            patch_size=16,
            device=device,
            kdata=kdata,
            root_path=input_dir,
            image_pairs_kapture=image_pairs,
            colmap_db=colmap_db,
            dense_matching=False,
            pixel_tol=5,
            conf_thr=1.001,
            skip_geometric_verification=False,
            min_len_track=3
        )
        colmap_db.close()
    except Exception as e:
        print(f"Error during reconstruction: {str(e)}")
        colmap_db.close()
        raise e
    if len(colmap_image_pairs) == 0:
        raise Exception("no matches were kept")

    ############################################ Verify matches and run reconstruction
    print("verify_matches")
    with open(os.path.join(output_dir, "pairs.txt"), "w") as f:
        for image_path1, image_path2 in colmap_image_pairs:
            f.write("{} {}\n".format(image_path1, image_path2))
    pycolmap.verify_matches(db_path, os.path.join(output_dir, "pairs.txt"))
    recon_path = os.path.join(output_dir, "reconstruction")
    if os.path.isdir(recon_path):
        shutil.rmtree(recon_path)
    os.makedirs(recon_path, exist_ok=True)
    pycolmap_run_mapper(db_path, recon_path, input_dir)

    ############################################ Save results
    # Load the reconstruction results from COLMAP
    reconstruction_folder = os.path.join(recon_path, "0")
    # Extract and save correspondences between 2D and 3D points
    correspondence_file = os.path.join(output_dir, "colmap_raw.json")
    correspondence_data = extract_and_save_correspondences(reconstruction_folder, 
                                                           correspondence_file)
    
    K, world2cam, points3D = process_colmap_data(correspondence_data)
    cam2world = np.asarray([inv(p) for p in world2cam])
    
    return K, cam2world, points3D


if __name__ == "__main__":
    run_mast3r(input_dir="./data/0318/imgs",
               output_dir="./output/0318",
               model_path="./mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth")
