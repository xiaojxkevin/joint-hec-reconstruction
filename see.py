import cv2
import numpy as np


def project_points_to_image(points, cam2world, intrinsics, image_width, image_height):
    """
    Projects 3D points (and their colors) from the world frame onto a 2D image.

    Args:
        points (np.ndarray): Array of shape (N, 3) containing the 3D world coordinates.
        colors (np.ndarray): Array of shape (N, 3) containing the RGB color of each point.
        cam2world (np.ndarray): 4x4 homogeneous transformation matrix for the camera pose in the world.
        intrinsics (np.ndarray): 3x3 intrinsic matrix.
        image_width (int): Width of the output image.
        image_height (int): Height of the output image.

    Returns:
        np.ndarray: An image (H x W x 3) with the points projected onto it.
    """

    # Create output image and a depth buffer for z-buffering.
    image = np.zeros((image_height, image_width, 3), dtype=np.float32)
    depth_buffer = np.full((image_height, image_width), np.inf)

    world2cam = np.linalg.inv(cam2world)
    R = world2cam[:3, :3]
    t = world2cam[:3, 3]

    num_points = points.shape[0]
    points_cam = points @ R.T + t  # (N, 3)

    # Extract coordinates.
    x = points_cam[:, 0]
    y = points_cam[:, 1]
    z = points_cam[:, 2]

    valid = z > 0
    x = x[valid]
    y = y[valid]
    z = z[valid]
    print(f"Valid points: {len(x)} with total number of points: {num_points}")

    # Project points onto the image plane using the intrinsic matrix.
    # u = (fx * x) / z + cx;  v = (fy * y) / z + cy
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]

    u = (fx * x) / z + cx
    v = (fy * y) / z + cy

    # Convert to integer pixel indices.
    u = np.round(u).astype(int)
    v = np.round(v).astype(int)

    # Loop through all valid points and update the image using a z-buffer.
    for ui, vi, depth in zip(u, v, z):
        # Check if the projected pixel is inside image boundaries.
        if 0 <= ui < image_width and 0 <= vi < image_height:
            if depth < depth_buffer[vi, ui]:
                depth_buffer[vi, ui] = depth
                image[vi, ui] = [255, 255, 255]

    return image, u, v


def load(data_path):
    data = np.load(data_path, allow_pickle=True)
    K = data["K"]
    eye2base_poses = data["eye2base"]
    points = data["pts_in_base"]
    visibility = data["visibility"].item()
    points2d = data["points2D"]
    return K, eye2base_poses, points, visibility, points2d


def proj_img(i, K, eye2base_poses, points, visibility, h, w):
    eye2base = eye2base_poses[i]
    indicies3d = visibility[i]["pts3d_indices"]
    sele_pts = points[indicies3d]
    img3d, u, v = project_points_to_image(
        points=sele_pts,
        cam2world=eye2base,
        intrinsics=K,
        image_width=w,
        image_height=h,
    )
    img3d = (np.clip(img3d, 0, 1) * 255).astype(np.uint8)
    return img3d


def main(expname):
    init_path = f"proj/{expname}-init.npz"
    final_path = f"proj/{expname}-final.npz"
    K, init_eye2base_poses, init_points, init_visibility, init_points2d = load(
        init_path
    )
    K, final_eye2base_poses, final_points, final_visibility, final_points2d = load(
        final_path
    )

    for i in range(init_eye2base_poses.shape[0]):
        init_img3d = proj_img(
            i, K, init_eye2base_poses, init_points, init_visibility, 480, 640
        )
        final_img3d = proj_img(
            i, K, final_eye2base_poses, final_points, final_visibility, 480, 640
        )

        pts2d = init_points2d[init_visibility[i]["pts2d_indices"]]
        img2d = np.zeros((480, 640, 3), dtype=np.uint8)
        for pt in pts2d:
            u, v = int(pt[0]), int(pt[1])
            if 0 <= u < 640 and 0 <= v < 480:
                img2d[v, u] = [0, 255, 0]

        stacked = np.vstack(
            (np.hstack((init_img3d, img2d)), np.hstack((final_img3d, img2d)))
        )

        cv2.imshow(f"{expname} [[init 3d, img2d],[ba 3d, img2d]]", stacked)
        key = cv2.waitKey(0)
        if key == 27:
            continue
        # cv2.imwrite(f"{i:02d}.png", cv2.cvtColor(img3d, cv2.COLOR_RGB2BGR))
    cv2.destroyAllWindows()


if __name__ == "__main__":
    expname = "0318_08"
    main(expname)
