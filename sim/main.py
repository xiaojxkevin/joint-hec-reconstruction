# Download chessboard object from https://github.com/DLR-RM/BlenderProc/blob/main/examples/advanced/calibration/README.md
import blenderproc as bproc
import numpy as np
import argparse
import os
import PIL.Image as Image
from scipy.spatial.transform import Rotation
import pathlib
from typing import List
import yaml


def random_se3(max_translation=0.1, max_rotation_deg=180):
    """
    Generate a random SE(3) transformation matrix (including rotation and translation).

    Args:
        max_translation: Maximum translation value in each axis.
        max_rotation_deg: Maximum rotation in degrees.

    Returns:
        np.ndarray: A 4x4 numpy array representing the SE(3) transformation.
    """
    # Generate a random translation vector
    t = np.random.uniform(-max_translation, max_translation, 3)

    # Generate a random rotation using scipy's Rotation
    theta = np.random.uniform(-max_rotation_deg, max_rotation_deg)

    # Random unit vector for rotation axis
    axis = np.random.randn(3)
    axis /= np.linalg.norm(axis)  # Normalize the rotation axis

    # Create rotation using scipy's Rotation
    rot = Rotation.from_rotvec(axis * np.radians(theta))
    R = rot.as_matrix()

    # Construct the SE(3) matrix
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def inv(T):
    """
    Inverse a SE(3) matrix.

    Args:
        T: 4x4 transformation matrix to invert.

    Returns:
        np.ndarray: Inverted 4x4 transformation matrix.
    """
    assert T.shape == (4, 4), f"Wrong shape: {T.shape}, should be (4, 4)"
    R = T[:3, :3]
    t = T[:3, 3]
    R_inv = R.T
    t_inv = -R_inv @ t
    T_inv = np.eye(4)
    T_inv[:3, :3] = R_inv
    T_inv[:3, 3] = t_inv
    return T_inv


class HandEyeCalibrationSimulator:
    def __init__(self, config_path: str, exp_name: str):
        """Initialize the hand-eye calibration simulator.

        Args:
            config_path: Path to the config file (.yaml)
        """
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)

        self.output_dir = os.path.join(config["output_dir"], exp_name)
        os.makedirs(self.output_dir, exist_ok=True)
        self.num_poses = config["num_poses"]
        self.num_objects = config["num_objects"]
        self.resolution = config["resolution"]
        self.camera_pose_method = config["cam_method"]
        self.size = config["size"]
        self.max_samples = config["max_sample"]
        self.SHAPENET_PATH = "/mnt/data/ShapeNetCoreV2/data"
        # self.categories = sorted(os.listdir(self.SHAPENET_PATH))
        self.categories = config["categories"]
        self.use_chessboard = config["use_chessboard"]
        self.K = None
        self.ground_plane = None
        self.mesh_objects: List[bproc.types.MeshObject] = []

        self.eye2hand_pose = random_se3(max_translation=0.1, max_rotation_deg=120)
        self.base2world_pose = np.eye(4)
        self.base2world_pose[:3, 3] = np.array([0.3, 0.4, 0.1])

        self.hand2base_poses = []
        self.eye2base_poses = []

    def setup_scene(self):
        """Initialize BlenderProc and set up the scene."""
        bproc.init()

        # Setup camera parameters
        bproc.camera.set_resolution(self.resolution["width"], self.resolution["height"])
        self.K = bproc.camera.get_intrinsics_as_K_matrix()
        print("Camera intrinsics:")
        print(self.K)

        # Create ground plane
        self.ground_plane, chessboard, protective_box = self.create_basic_scene()
        if chessboard is not None:
            self.mesh_objects.append(chessboard)
            self.mesh_objects.append(protective_box)
        else:
            self.num_objects += 1

        # Create objects
        while True:
            obj = self.load_single_object()
            if obj is not None:
                self.mesh_objects.append(obj)
            if len(self.mesh_objects) >= self.num_objects:
                break
        print("<>" * 20)
        print(f"Finished loading {len(self.mesh_objects)} objects")

        # Add multiple light sources to reduce shadows
        self.setup_lighting()

    def create_basic_scene(self):
        # Create a plane as ground plane
        ground_plane = bproc.object.create_primitive("PLANE", scale=[7, 7, 1])
        color = (0.65, 0.45, 0.35, 1.0)  # Brown color
        ground_material = bproc.material.create("GroundMaterial")
        ground_material.set_principled_shader_value("Base Color", list(color))
        ground_material.set_principled_shader_value("Roughness", 0.7)
        ground_plane.replace_materials(ground_material)

        # place a chessboard
        chessboard, protective_box = None, None
        if self.use_chessboard:
            chessboard = bproc.loader.load_obj("./chess.obj")[0]
            bounding_box = chessboard.get_bound_box()
            min_x, min_y, min_z = np.min(bounding_box, axis=0)
            max_x, max_y, max_z = np.max(bounding_box, axis=0)
            width = max_x - min_x
            length = max_z - min_z
            # Rescaling
            current_max_dimension = max(width, length)
            # The chessboard has 7x8 grids, let's make the the size of each gride to 0.1
            required_scale = 0.8 / current_max_dimension
            chessboard.set_scale([required_scale, required_scale, required_scale])
            chessboard.set_rotation_euler([0, 0, np.pi / 2])

            updated_bounding_box = chessboard.get_bound_box()
            min_x, min_y, min_z = np.min(updated_bounding_box, axis=0)
            max_x, max_y, max_z = np.max(updated_bounding_box, axis=0)

            # Create an invisible box slightly larger than the chessboard's footprint
            # and positioned above it
            box_height = 0.8
            protective_box = bproc.object.create_primitive("CUBE")

            # Scale the box to match chessboard dimensions with a small margin
            margin = 0.01  # Small margin around the edges
            box_width = (max_x - min_x) + 2 * margin
            box_length = (max_z - min_z) + 2 * margin
            protective_box.set_scale([box_width / 2, box_length / 2, box_height / 2])

            # Position the box above the chessboard
            center_x = (min_x + max_x) / 2
            center_z = (min_z + max_z) / 2
            box_y_position = (
                max_y + box_height / 2
            )  # Place box directly above chessboard
            protective_box.set_location([center_x, box_y_position, center_z])

            protective_box.hide(True)
            protective_box.enable_rigidbody(True, collision_shape="BOX", mass=0)

        return ground_plane, chessboard, protective_box

    def load_single_object(self):
        # Load a single object from ShapeNet for the scene
        selected_synset = np.random.choice(self.categories)
        category_path = os.path.join(self.SHAPENET_PATH, selected_synset)
        model_ids = [
            d
            for d in os.listdir(category_path)
            if os.path.isdir(os.path.join(category_path, d))
        ]
        if not model_ids:
            raise ValueError(f"No models found in synset category {selected_synset}")
        selected_model = np.random.choice(model_ids)
        model_path = os.path.join(
            category_path, selected_model, "models/model_normalized.obj"
        )
        objs = bproc.loader.load_obj(model_path)
        # objs = bproc.loader.load_obj("ShapeNetCoreV2/data/02691156/1a04e3eab45ca15dd86060f189eb133/models/model_normalized.obj")
        assert len(objs) == 1, f"Expected 1 object, got {len(objs)}"
        print(f"Loaded model: {selected_model} from category {selected_synset}")
        obj = objs[0]
        min_x, min_y, min_z = np.min(obj.get_bound_box(), axis=0)

        # Add the object to the scene
        from blenderproc.python.utility.CollisionUtility import CollisionUtility

        max_attempts = 8
        is_valid = False
        for attempt in range(max_attempts):
            position_x = np.random.uniform(-self.size, self.size)
            position_y = np.random.uniform(-self.size, self.size)
            position_z = -min_z
            # TODO: fix the problem of rotation
            rotation_z = np.random.uniform(0, 2 * np.pi)
            # obj.set_rotation_euler([0, 0, rotation_z])
            obj.set_location([position_x, position_y, position_z])

            is_valid = CollisionUtility.check_intersections(
                obj=obj,
                bvh_cache=None,
                objects_to_check_against=self.mesh_objects,
                list_of_objects_with_no_inside_check=[],
            )
            if is_valid:
                break

        if is_valid:
            print(
                f"Placed object at position [{position_x:.2f}, {position_y:.2f}, {position_z:.2f}]."
            )
            return obj
        print("*************Failed to place object after maximum attempts*************")
        return None  # None if the object could not be placed after max attempts

    def setup_lighting(self):
        """Set up a comprehensive lighting system to minimize shadows."""
        # Create a key light (main light source)
        key_light = bproc.types.Light()
        key_light.set_type("POINT")
        key_light.set_location([5, -5, 5])
        key_light.set_energy(500)

        # Create a fill light (reduces shadows from the key light)
        fill_light = bproc.types.Light()
        fill_light.set_type("POINT")
        fill_light.set_location([-5, 5, 5])
        fill_light.set_energy(500)

        # Add back light (creates separation between objects and background)
        back_light = bproc.types.Light()
        back_light.set_type("POINT")
        back_light.set_location([0, -5, 5])
        back_light.set_energy(500)

        # Add rim light (highlights object edges)
        rim_light = bproc.types.Light()
        rim_light.set_type("POINT")
        rim_light.set_location([-5, -5, 5])
        rim_light.set_energy(500)

        # Add overhead fill light
        overhead_light = bproc.types.Light()
        overhead_light.set_type("POINT")
        overhead_light.set_location([0, 0, 8])
        overhead_light.set_energy(500)

        # Add additional corner lights for even coverage
        corner_light1 = bproc.types.Light()
        corner_light1.set_type("POINT")
        corner_light1.set_location([5, 5, 5])
        corner_light1.set_energy(300)

        corner_light2 = bproc.types.Light()
        corner_light2.set_type("POINT")
        corner_light2.set_location([-2, 2, 3])
        corner_light2.set_energy(300)

        corner_light3 = bproc.types.Light()
        corner_light3.set_type("POINT")
        corner_light3.set_location([2, -2, 3])
        corner_light3.set_energy(300)

        # Create ambient lighting (affects the entire scene)
        ambient_light = bproc.types.Light()
        ambient_light.set_type("SUN")
        ambient_light.set_location([0, 0, 10])
        ambient_light.set_rotation_euler([0, 0, 0])
        ambient_light.set_energy(0.5)

        # Add a ground bounce light (simulates light reflected from ground)
        bounce_light = bproc.types.Light()
        bounce_light.set_type("AREA")
        bounce_light.set_location([0, 0, 0.1])
        bounce_light.set_rotation_euler([np.pi, 0, 0])  # Point upward
        bounce_light.set_energy(100)
        bounce_light.set_scale([4, 4, 1])  # Wide area light

    def generate_camera_poses(self):
        """Generate camera poses and corresponding robot hand poses based on the selected method.

        Args:
            objs: List of scene objects.
        """
        # Create BVH tree for collision detection
        bvh_tree = bproc.object.create_bvh_tree_multi_objects(self.mesh_objects)

        # Find point of interest, all cam poses should look towards it
        poi = bproc.object.compute_poi(self.mesh_objects)

        if self.camera_pose_method == "random":
            self._generate_random_camera_poses(bvh_tree, poi, self.num_poses)
        elif self.camera_pose_method == "circle":
            self._generate_circular_camera_poses(bvh_tree, poi)
        else:
            raise ValueError(f"Unknown camera pose method: {self.camera_pose_method}")

    def _generate_random_camera_poses(self, bvh_tree, poi, num_poses):
        """Generate random camera poses around the scene.

        Args:
            objs: List of scene objects.
            bvh_tree: BVH tree for collision detection.
            poi: Point of interest to look at.
        """
        generated_poses = 0
        tries = 0

        while tries < 1000 and generated_poses < num_poses:
            tries += 1

            # Sample random camera location above objects
            offset = 0.8
            location = np.random.uniform(
                [-self.size - offset, -self.size - offset, self.size * 2.0 - 0.2],
                [self.size + offset, self.size + offset, self.size * 2.0 + 0.2],
            )

            # Random in-plane rotation (around viewing direction)
            max_deg = 90
            inplane_rot = np.random.uniform(-np.radians(max_deg), np.radians(max_deg))

            # Add camera pose if it passes all checks
            if self._add_camera_pose(location, inplane_rot, poi, bvh_tree):
                generated_poses += 1

    def _generate_circular_camera_poses(self, bvh_tree, poi):
        """Generate camera poses in a circular pattern above the scene.

        Args:
            objs: List of scene objects.
            bvh_tree: BVH tree for collision detection.
            poi: Point of interest to look at.
        """
        # Get ellipse parameters from camera_pose_params or use defaults
        radius = self.size + 0.6
        height = self.size * 2.0
        center = poi.copy()

        # Generate evenly spaced angles
        angles = np.linspace(0, 2 * np.pi, self.num_poses, endpoint=False)

        generated_poses = 0
        for angle in angles:
            # Calculate camera position on the ellipse
            x = center[0] + radius * np.cos(angle)
            y = center[1] + radius * np.sin(angle)
            z = height
            location = np.array([x, y, z]) + np.random.uniform(-0.4, 0.4, 3)

            # Random in-plane rotation for variety
            max_deg = 90  # Limit in-plane rotation to avoid extreme angles
            inplane_rot = np.random.uniform(-np.radians(max_deg), np.radians(max_deg))

            # Add camera pose if it passes all checks
            if self._add_camera_pose(location, inplane_rot, poi, bvh_tree):
                generated_poses += 1

        # If we couldn't generate enough poses, fill in with random ones
        if generated_poses < self.num_poses:
            print(
                f"Could only generate {generated_poses} poses. Filling with random poses."
            )
            self._generate_random_camera_poses(
                bvh_tree, poi, self.num_poses - generated_poses
            )

    def _add_camera_pose(self, location, inplane_rot, poi, bvh_tree):
        """Add a camera pose if it passes visibility checks.

        Args:
            location: Camera location.
            inplane_rot: In-plane rotation.
            poi: Point of interest to look at.
            bvh_tree: BVH tree for collision detection.

        Returns:
            bool: True if the pose was added, False otherwise.
        """
        # Compute rotation based on vector going from location towards poi
        rotation_matrix = bproc.camera.rotation_from_forward_vec(
            poi - location, inplane_rot=inplane_rot
        )

        # Build transformation matrix
        cam2world_matrix = bproc.math.build_transformation_mat(
            location, rotation_matrix
        )

        # Check for obstacles in the camera view
        if not bproc.camera.perform_obstacle_in_view_check(
            cam2world_matrix, {"min": 1.0}, bvh_tree
        ):
            return False

        # Add the camera pose
        bproc.camera.add_camera_pose(cam2world_matrix)

        # Calculate the corresponding hand2base and eye2base poses
        hand2base, eye2base = self.calculate_hand2base_from_camera_pose(
            cam2world_matrix, self.eye2hand_pose
        )
        self.hand2base_poses.append(hand2base)
        self.eye2base_poses.append(eye2base)

        return True

    def calculate_hand2base_from_camera_pose(self, cam2world, eye2hand):
        # Convert from camera to hand (end-effector)
        cam2world[:3, 1:3] *= -1  # Flip y and z axes for Blender's coordinate system
        hand2world = cam2world @ inv(eye2hand)

        # Calculate hand2base
        hand2base = inv(self.base2world_pose) @ hand2world
        eye2base = inv(self.base2world_pose) @ cam2world

        return hand2base, eye2base

    def save_intrinsics(self):
        """Save camera intrinsics to a txt file."""
        intrinsics_path = os.path.join(self.output_dir, "intrinsics.txt")
        np.savetxt(intrinsics_path, self.K, fmt="%.6f")

    def save_eye2hand_pose(self):
        """Save ground truth eye2hand poses to txt files."""
        pose_path = os.path.join(self.output_dir, f"eye2hand_pose.txt")
        np.savetxt(pose_path, self.eye2hand_pose, fmt="%.6f")

    def save_hand2base_poses_tum(self):
        """Save hand2base poses in TUM format to a single txt file."""
        tum_path = os.path.join(self.output_dir, "hand_tum.txt")
        with open(tum_path, "w") as f:
            for i, pose in enumerate(self.hand2base_poses):
                # Extract translation
                tx, ty, tz = pose[:3, 3]

                # Extract rotation matrix and convert to quaternion
                R = pose[:3, :3]
                rot = Rotation.from_matrix(R)
                qx, qy, qz, qw = rot.as_quat()  # Returns x, y, z, w

                # Use frame index as timestamp
                timestamp = float(i)

                # Write to file in TUM format
                f.write(
                    f"{timestamp:.6f} {tx:.6f} {ty:.6f} {tz:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n"
                )

    def save_eye2base_poses_tum(self):
        """Save eye2base poses in TUM format to a single txt file."""
        tum_path = os.path.join(self.output_dir, "eye_tum.txt")
        with open(tum_path, "w") as f:
            for i, pose in enumerate(self.eye2base_poses):
                # Extract translation
                tx, ty, tz = pose[:3, 3]

                # Extract rotation matrix and convert to quaternion
                R = pose[:3, :3]
                rot = Rotation.from_matrix(R)
                qx, qy, qz, qw = rot.as_quat()
                timestamp = float(i)
                f.write(
                    f"{timestamp:.6f} {tx:.6f} {ty:.6f} {tz:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n"
                )

    def save_images(self, rendered_data):
        """Save RGB, depth and normal images with sequential numbering.

        Args:
            rendered_data: Dictionary containing rendered image data.
        """
        # Extract RGB images
        img_dir = os.path.join(self.output_dir, "imgs")
        os.makedirs(img_dir, exist_ok=True)
        if "colors" in rendered_data:
            rgb_images = rendered_data["colors"]

            for i, img_array in enumerate(rgb_images):
                # Format index with leading zero
                index = f"{i+1:02d}"
                img_path = os.path.join(img_dir, f"{index}.png")

                # Convert numpy array to PIL Image and save
                img_array_uint8 = img_array.astype(np.uint8)
                img = Image.fromarray(img_array_uint8)
                img.save(img_path)

    def render_and_save(self):
        """Render the scene and save all data in the required format."""
        # Set renderer parameters for better quality
        bproc.renderer.set_noise_threshold(0.01)
        bproc.renderer.set_max_amount_of_samples(self.max_samples)
        # bproc.renderer.set_denoiser("")

        # Render the scene
        data = bproc.renderer.render()

        # Save RGB images
        self.save_images(data)

        # Save camera intrinsics
        self.save_intrinsics()

        # Save ground truth eye2hand pose
        self.save_eye2hand_pose()
        self.save_eye2base_poses_tum()

        # Save hand2base poses in TUM format
        self.save_hand2base_poses_tum()

    def run_simulation(self):
        """Run the full simulation pipeline."""
        # Setup scene
        self.setup_scene()

        # Generate camera and robot poses
        self.generate_camera_poses()

        # Render and save data
        self.render_and_save()

        return self.output_dir


def main():
    """Main function to run the hand-eye calibration simulation."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config_path",
        nargs="?",
        help="Path to the config file",
    )
    parser.add_argument(
        "exp_name",
        nargs="?",
    )
    args = parser.parse_args()

    # Create the simulator
    simulator = HandEyeCalibrationSimulator(
        config_path=args.config_path, exp_name=args.exp_name
    )

    # Run the simulation
    output_dir = simulator.run_simulation()
    print(f"Hand-eye calibration simulation completed. Data saved to {output_dir}")


if __name__ == "__main__":
    main()
