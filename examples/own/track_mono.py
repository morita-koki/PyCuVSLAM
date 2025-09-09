#
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
import os
import yaml
from typing import List

import numpy as np
from PIL import Image
import rerun as rr
import rerun.blueprint as rrb
import cv2

import cuvslam

# Set up dataset path
# config_path = os.path.join(os.path.dirname(__file__), "setting.yaml")
# config_path = os.path.join(os.path.dirname(__file__), "setting_metro_20241227.yaml")




def load_frame(image_path: str, decompressor) -> np.ndarray:
    """Load an image from a file path."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    

    if 'bayer' in decompressor.format_from:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    elif 'rgb' in decompressor.format_from:
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
    else:
        raise ValueError(f"Unsupported encoding format: {decompressor.format_from}")

    return decompressor.decompress(img)

    # image = Image.open(image_path)
    # frame = np.array(image)

    # if image.mode == 'L':
    #     # mono8
    #     if len(frame.shape) != 2:
    #         raise ValueError("Expected mono8 image to have 2 dimensions [H W].")
    # elif image.mode == 'RGB':
    #     # rgb8 - convert to BGR for cuvslam compatibility
    #     if len(frame.shape) != 3 or frame.shape[2] != 3:
    #         raise ValueError(
    #             "Expected rgb8 image to have 3 dimensions with 3 channels [H W C].")
    #     # Convert RGB to BGR by reversing the channel order and ensure contiguous
    #     frame = np.ascontiguousarray(frame[:, :, ::-1])
    # elif image.mode == 'I;16':
    #     # uint16 depth image
    #     if len(frame.shape) != 2:
    #         raise ValueError("Expected uint16 depth image to have 2 dimensions [H W].")
    #     frame = frame.astype(np.uint16)
    # else:
    #     raise ValueError(f"Unsupported image mode: {image.mode}")

    # return frame

def _load_yaml_config(yaml_path: str) -> dict:
    """Load YAML configuration file."""
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Sensor YAML not found: {yaml_path}")
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)


import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))      
from utils.image.decompress import ImageDecompressor
def get_decompressor(config_path: str) -> ImageDecompressor:
    """Get an ImageDecompressor based on the configuration file."""
    config = _load_yaml_config(config_path)
    encode = config.get("encode", None)
    if encode is None:
        raise ValueError("Missing 'encode' field in configuration.")
    return ImageDecompressor(encode)

def _create_camera_from_config(config: dict) -> cuvslam.Camera:
    """Create camera object from config."""
    cam = cuvslam.Camera()
    cam.focal = [config['intrinsic'][0], config['intrinsic'][4]]
    cam.principal = [config['intrinsic'][2], config['intrinsic'][5]]
    cam.size = config['resolution']
    
    # Brown-Conrady distortion model requires only k1, k2, k3, p1, p2
    # but only k1, k2, p1, p2 are provided in the default calibration
    cam.distortion = cuvslam.Distortion(
        cuvslam.Distortion.Model.Brown,
        config['distortion'][:2] + [config['distortion'][4]] + config['distortion'][2:4]
    )

    return cam


def _get_sensor_path(path: str) -> str:
    return os.path.join(path, 'camera_front.yaml')


def prepare_frame_metadata(config_path: str) -> List[dict]:
    """Process EuRoC dataset camera files and generate synchronized frame metadata."""
    config = _load_yaml_config(config_path)
    workdir = config.get("workdir", ".")
    if not os.path.exists(workdir):
        raise ValueError(f"EuRoC dataset path does not exist: {workdir}")

    # left_cam_csv = os.path.join(euroc_path, 'cam0', 'data.csv')

    front_cameras = [path for path in os.listdir(os.path.join(workdir, 'camera', 'front')) if path.endswith('.jpg')]

    timestamps = [os.path.splitext(name)[0] for name in front_cameras]

    # Generate stereo frame metadata
    frames_metadata = [{
        'type': 'mono',
        'timestamp': timestamp,
        'images_paths': [
            os.path.join(workdir, 'camera', 'front', f"{timestamp}.jpg"),
        ],
    } for timestamp in timestamps]

    # Sort frames by timestamp
    frames_metadata.sort(key=lambda x: x['timestamp'])

    return frames_metadata

def get_rig(config_path: str) -> cuvslam.Rig:
    """Get a Rig object from EuRoC dataset path with transformations relative to cam0."""
    # cam0_path  = _get_sensor_path(path)
    
    # Load configurations
    config = _load_yaml_config(config_path)
    camera_configs = config.get("cameras", [])
    # nameがfrontのものを探す
    front_camera_config = next((cam for cam in camera_configs if cam.get("name") == "front"), None)
    print(front_camera_config)
    # front_camera_config = _load_yaml_config(config_path)
    
    # Create cameras
    front_camera = _create_camera_from_config(front_camera_config)
    
    # front_camera becomes identity
    front_camera.rig_from_camera = cuvslam.Pose(
        rotation=[0, 0, 0, 1],  # Identity quaternion
        translation=[0, 0, 0]    # Zero translation
    )
    
    rig = cuvslam.Rig()
    rig.cameras = [front_camera]
    
    return rig


def color_from_id(identifier):
    """Generate pseudo-random color from integer identifier for visualization."""
    return [
        (identifier * 17) % 256,
        (identifier * 31) % 256,
        (identifier * 47) % 256
    ]

def argument_parser():
    import argparse
    parser = argparse.ArgumentParser(
        description="Example of using cuVSLAM for visual odometry with EuRoC dataset."
    )
    parser.add_argument(
        "config_path",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "setting.yaml"),
        help="Path to the YAML configuration file.",
    )
    return parser.parse_args()

def main():
    args = argument_parser()
    config_path = args.config_path
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    # Setup rerun visualizer
    rr.init("cuVSLAM Visualizer", spawn=True)

    # Setup coordinate basis for root, cuvslam uses right-hand system with
    # X-right, Y-down, Z-forward
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)
    # Setup rerun views
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.TimePanel(state="collapsed"),
            rrb.Horizontal(
                column_shares=[0.5, 0.5],
                contents=[
                    rrb.Vertical(contents=[
                        rrb.Horizontal(contents=[
                            rrb.Spatial2DView(origin='world/front'),
                        ]),
                        ])
                    ]),
                    rrb.Spatial3DView(origin='world')
            )
        )

    # Available tracking modes:
    # 0: Multicamera - Visual tracking using stereo camera (can be extended to multiple stereo cameras)
    # 1: Inertial - Visual-inertial tracking using stereo camera + IMU
    # 2: RGBD - Visual tracking using monocular camera + depth (supports grayscale input)
    # 3: Mono - Visual tracking using monocular camera (without scale, accurate rotation only)

    tracking_mode = cuvslam.Tracker.OdometryMode(3)

    # Configure tracker
    cfg = cuvslam.Tracker.OdometryConfig(
        async_sba=False,
        enable_observations_export=True,
        enable_final_landmarks_export=True,
        horizontal_stereo_camera=False,
        odometry_mode=tracking_mode
    )
    
    # Get decompressor
    decompressor = get_decompressor(config_path)

    # Get camera rig
    rig = get_rig(config_path)

    # Initialize tracker
    tracker = cuvslam.Tracker(rig, cfg)
    print(f"cuVSLAM Tracker initilized with odometry mode: {cfg.odometry_mode}")

    # Track frames
    last_camera_timestamp = None
    frame_id = 0
    trajectory = []
    frames_metadata = prepare_frame_metadata(
        config_path
    )

    odom_trajectory = []

    for i, frame_metadata in enumerate(frames_metadata):
        # if i < 1500: continue  # Skip first 1500 frames for metro_20241227
        timestamp = frame_metadata['timestamp']
        
        images = [load_frame(image_path,decompressor) for image_path in frame_metadata['images_paths']]

        # Reset counters
        last_camera_timestamp = timestamp

        # Track frame
        odom_pose_estimate, _ = tracker.track(timestamp, images)

        if odom_pose_estimate.world_from_rig is None:
            print(f"Warning: Failed to track frame {frame_id}")
            continue

        # Get current pose and observations for the main camera and gravity in rig frame
        odom_pose = odom_pose_estimate.world_from_rig.pose
        current_observations_main_cam = tracker.get_last_observations(0)
        trajectory.append(odom_pose.translation)
        odom_trajectory.append([timestamp] + list(odom_pose.translation) + list(odom_pose.rotation))

        # gravity = None
        # if cfg.odometry_mode == cuvslam.Tracker.OdometryMode.Inertial:
        #     # Gravity estimation requires collecting sufficient number of keyframes with motion diversity
        #     gravity = tracker.get_last_gravity()

        # Visualize
        rr.set_time_sequence("frame", frame_id)
        rr.log("world/trajectory", rr.LineStrips3D(trajectory), static=True)
        rr.log(
            "world/front",
            rr.Transform3D(
                translation=odom_pose.translation,
                quaternion=odom_pose.rotation
            ),
            rr.Arrows3D(
                vectors=np.eye(3) * 0.2,
                colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]]  # RGB for XYZ axes
            )
        )

        points = np.array([[obs.u, obs.v] for obs in current_observations_main_cam])
        colors = np.array([color_from_id(obs.id) for obs in current_observations_main_cam])
        rr.log(
            "world/front/observations",
            rr.Points2D(positions=points, colors=colors, radii=5.0),
            rr.Image(images[0]).compress(jpeg_quality=80)
        )



        frame_id += 1


if __name__ == "__main__":
    main()  