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
import queue
import threading
from copy import deepcopy
from typing import List, Optional

import numpy as np
import pyrealsense2 as rs

import cuvslam as vslam
from camera_utils import get_rs_vio_rig
from visualizer import RerunVisualizer

# Constants
RESOLUTION = (640, 360)
FPS = 30
IMU_FREQUENCY_ACCEL = 250
IMU_FREQUENCY_GYRO = 200
IMAGE_JITTER_THRESHOLD_MS = 35 * 1e6  # 35ms in nanoseconds
IMU_JITTER_THRESHOLD_MS = 6 * 1e6  # 6ms in nanoseconds


class ThreadWithTimestamp:
    """Helper class to manage timestamps between camera and IMU threads."""
    
    def __init__(
        self,
        low_rate_threshold_ns: int,
        high_rate_threshold_ns: int
    ) -> None:
        """Initialize timestamp tracker.
        
        Args:
            low_rate_threshold_ns: Threshold for low-rate (camera) stream
            high_rate_threshold_ns: Threshold for high-rate (IMU) stream
        """
        self.prev_low_rate_timestamp: Optional[int] = None
        self.prev_high_rate_timestamp: Optional[int] = None
        self.low_rate_threshold_ns = low_rate_threshold_ns
        self.high_rate_threshold_ns = high_rate_threshold_ns
        self.last_low_rate_timestamp: Optional[int] = None


def imu_thread(
    tracker: vslam.Tracker,
    q: queue.Queue,
    thread_with_timestamp: ThreadWithTimestamp,
    motion_pipe: rs.pipeline
) -> None:
    """IMU processing thread.
    
    Args:
        tracker: cuVSLAM tracker instance
        q: Queue for communication with main thread
        thread_with_timestamp: Timestamp management object
        motion_pipe: RealSense motion pipeline
    """
    try:
        while True:
            imu_measurement = vslam.ImuMeasurement()
            imu_frames = motion_pipe.wait_for_frames()
            current_timestamp = int(imu_frames[0].timestamp * 1e6)

            # Check timestamp consistency with camera thread
            if (thread_with_timestamp.last_low_rate_timestamp is not None and
                    current_timestamp < thread_with_timestamp.last_low_rate_timestamp):
                print(
                    f"Warning: IMU stream timestamp is earlier than camera "
                    f"stream ({current_timestamp} < "
                    f"{thread_with_timestamp.last_low_rate_timestamp})"
                )
                continue

            # Check for timestamp gaps in IMU stream
            timestamp_diff = 0
            if thread_with_timestamp.prev_high_rate_timestamp is not None:
                timestamp_diff = (
                    current_timestamp - thread_with_timestamp.prev_high_rate_timestamp
                )
                if timestamp_diff > thread_with_timestamp.high_rate_threshold_ns:
                    print(
                        f"Warning: IMU stream message drop: timestamp gap "
                        f"({timestamp_diff/1e6:.2f} ms) exceeds threshold "
                        f"{thread_with_timestamp.high_rate_threshold_ns/1e6:.2f} ms"
                    )
                elif timestamp_diff < 0:
                    print("Warning: IMU messages are not sequential")

            if timestamp_diff < 0:
                continue

            thread_with_timestamp.prev_high_rate_timestamp = deepcopy(
                current_timestamp
            )
            
            # Populate IMU measurement
            imu_measurement.timestamp_ns = current_timestamp
            accel_data = np.frombuffer(imu_frames[0].get_data(), dtype=np.float32)
            gyro_data = np.frombuffer(imu_frames[1].get_data(), dtype=np.float32)
            imu_measurement.linear_accelerations = accel_data[:3]
            imu_measurement.angular_velocities = gyro_data[:3]

            if timestamp_diff > 0:
                tracker.register_imu_measurement(0, imu_measurement)
    except Exception as e:
        print(f"IMU thread error: {e}")


def camera_thread(
    tracker: vslam.Tracker,
    q: queue.Queue,
    thread_with_timestamp: ThreadWithTimestamp,
    ir_pipe: rs.pipeline
) -> None:
    """Camera processing thread.
    
    Args:
        tracker: cuVSLAM tracker instance
        q: Queue for communication with main thread
        thread_with_timestamp: Timestamp management object
        ir_pipe: RealSense infrared pipeline
    """
    try:
        while True:
            ir_frames = ir_pipe.wait_for_frames()
            ir_left_frame = ir_frames.get_infrared_frame(1)
            ir_right_frame = ir_frames.get_infrared_frame(2)
            current_timestamp = int(ir_left_frame.timestamp * 1e6)

            # Check for timestamp gaps in camera stream
            if thread_with_timestamp.prev_low_rate_timestamp is not None:
                timestamp_diff = (
                    current_timestamp - thread_with_timestamp.prev_low_rate_timestamp
                )
                if timestamp_diff > thread_with_timestamp.low_rate_threshold_ns:
                    print(
                        f"Warning: Camera stream message drop: timestamp gap "
                        f"({timestamp_diff/1e6:.2f} ms) exceeds threshold "
                        f"{thread_with_timestamp.low_rate_threshold_ns/1e6:.2f} ms"
                    )

            thread_with_timestamp.prev_low_rate_timestamp = deepcopy(
                current_timestamp
            )
            
            images = (
                np.asanyarray(ir_left_frame.get_data()),
                np.asanyarray(ir_right_frame.get_data())
            )

            odom_pose_estimate, slam_pose = tracker.track(current_timestamp, images)
            odom_pose = odom_pose_estimate.world_from_rig.pose

            # Put result in queue for main thread
            q.put([current_timestamp, odom_pose, slam_pose, images])
            thread_with_timestamp.last_low_rate_timestamp = current_timestamp
    except Exception as e:
        print(f"Camera thread error: {e}")


def setup_camera_parameters() -> dict:
    """Set up camera parameters by starting pipeline briefly.
    
    Returns:
        Dictionary containing camera parameters
    """
    # Initialize RealSense configuration
    config = rs.config()
    pipeline = rs.pipeline()

    # Configure streams for initial setup
    config.enable_stream(
        rs.stream.infrared, 1, RESOLUTION[0], RESOLUTION[1], rs.format.y8, FPS
    )
    config.enable_stream(
        rs.stream.infrared, 2, RESOLUTION[0], RESOLUTION[1], rs.format.y8, FPS
    )
    config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, IMU_FREQUENCY_ACCEL)
    config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, IMU_FREQUENCY_GYRO)

    # Start pipeline to get intrinsics and extrinsics
    profile = pipeline.start(config)
    frames = pipeline.wait_for_frames()
    pipeline.stop()

    # Prepare camera parameters
    camera_params = {'left': {}, 'right': {}, 'imu': {}}

    # Get extrinsics and intrinsics
    camera_params['right']['extrinsics'] = frames[1].profile.get_extrinsics_to(
        frames[0].profile
    )
    camera_params['imu']['cam_from_imu'] = frames[2].profile.get_extrinsics_to(
        frames[0].profile
    )
    camera_params['left']['intrinsics'] = (
        frames[0].profile.as_video_stream_profile().intrinsics
    )
    camera_params['right']['intrinsics'] = (
        frames[1].profile.as_video_stream_profile().intrinsics
    )

    return camera_params

from scipy.spatial.transform import Rotation as R
# Lambda to convert quaternion [x, y, z, w] to 3x3 rotation matrix (as list of lists)
quaternion_to_rotation_matrix = lambda q: R.from_quat(q).as_matrix().tolist()

# Lambda to multiply two quaternions [x, y, z, w] * [x, y, z, w]
quaternion_multiply = lambda q1, q2: (R.from_quat(q1) * R.from_quat(q2)).as_quat().tolist()

# Lambda to rotate a 3D vector using a 3x3 rotation matrix
rotate_vector = lambda vector, rotation_matrix: R.from_matrix(rotation_matrix).apply(vector).tolist()


def combine_poses(initial_pose, relative_pose):
    """
    Combine initial pose with relative pose to get absolute pose.
    
    Args:
        initial_pose: cuvslam.Pose object representing initial pose
        relative_pose: cuvslam.Pose object representing relative pose
    
    Returns:
        cuvslam.Pose object representing combined absolute pose
    """
    # Get rotation matrix from initial pose quaternion
    rotation_matrix = quaternion_to_rotation_matrix(initial_pose.rotation)
    
    # Rotate relative translation by initial pose rotation
    rotated_rel_t = rotate_vector(relative_pose.translation, rotation_matrix)
    
    # Add initial translation
    absolute_translation = [
        initial_pose.translation[0] + rotated_rel_t[0],
        initial_pose.translation[1] + rotated_rel_t[1],
        initial_pose.translation[2] + rotated_rel_t[2]
    ]
    
    # Multiply quaternions
    absolute_rotation = quaternion_multiply(initial_pose.rotation, relative_pose.rotation)
    
    return vslam.Pose(translation=absolute_translation, rotation=absolute_rotation)


def transform_landmarks(landmarks, initial_pose):
    """
    Transform landmarks by initial pose (rotation + translation).
    
    Args:
        landmarks: list of 3D landmark coordinates
        initial_pose: cuvslam.Pose object representing initial pose
    
    Returns:
        List of transformed 3D landmark coordinates
    """
    rotation_matrix = quaternion_to_rotation_matrix(initial_pose.rotation)
    transformed_landmarks = []
    
    for landmark in landmarks:
        # Rotate landmark by initial pose rotation
        rotated_landmark = rotate_vector(landmark, rotation_matrix)
        
        # Add initial translation
        transformed_landmark = [
            initial_pose.translation[0] + rotated_landmark[0],
            initial_pose.translation[1] + rotated_landmark[1],
            initial_pose.translation[2] + rotated_landmark[2]
        ]
        transformed_landmarks.append(transformed_landmark)
    
    return transformed_landmarks


from numpy import array_equal as np_array_equal
def main() -> None:
    """Main function for VIO tracking."""
    # Setup camera parameters
    camera_params = setup_camera_parameters()

    # Configure tracker
    cfg = vslam.Tracker.OdometryConfig(
        async_sba=False,
        enable_final_landmarks_export=True,
        debug_imu_mode=False,
        odometry_mode=vslam.Tracker.OdometryMode.Inertial,
        horizontal_stereo_camera=True
    )
    SLAM_SYNC_MODE=True
    s_cfg = vslam.Tracker.SlamConfig(sync_mode=SLAM_SYNC_MODE)

    # Create rig using utility function
    rig = get_rs_vio_rig(camera_params)

    # Initialize tracker
    tracker = vslam.Tracker(rig, cfg, s_cfg)

    # Set up IR pipeline
    ir_pipe = rs.pipeline()
    ir_config = rs.config()
    ir_config.enable_stream(
        rs.stream.infrared, 1, RESOLUTION[0], RESOLUTION[1], rs.format.y8, FPS
    )
    ir_config.enable_stream(
        rs.stream.infrared, 2, RESOLUTION[0], RESOLUTION[1], rs.format.y8, FPS
    )

    # Configure device settings
    config_temp = rs.config()
    ir_wrapper = rs.pipeline_wrapper(ir_pipe)
    ir_profile = config_temp.resolve(ir_wrapper)
    device = ir_profile.get_device()

    # Disable IR emitter if supported
    depth_sensor = device.query_sensors()[0]
    if depth_sensor.supports(rs.option.emitter_enabled):
        depth_sensor.set_option(rs.option.emitter_enabled, 0)

    # Set up motion pipeline
    motion_pipe = rs.pipeline()
    motion_config = rs.config()
    motion_config.enable_stream(
        rs.stream.accel, rs.format.motion_xyz32f, IMU_FREQUENCY_ACCEL
    )
    motion_config.enable_stream(
        rs.stream.gyro, rs.format.motion_xyz32f, IMU_FREQUENCY_GYRO
    )

    # Start pipelines
    motion_pipe.start(motion_config)
    ir_pipe.start(ir_config)

    # Set up threading and visualization
    q = queue.Queue()
    visualizer = RerunVisualizer(num_viz_cameras=2)
    thread_with_timestamp = ThreadWithTimestamp(
        IMAGE_JITTER_THRESHOLD_MS, IMU_JITTER_THRESHOLD_MS
    )

    # Start threads
    imu_thread_obj = threading.Thread(
        target=imu_thread,
        args=(tracker, q, thread_with_timestamp, motion_pipe),
        daemon=True
    )
    camera_thread_obj = threading.Thread(
        target=camera_thread,
        args=(tracker, q, thread_with_timestamp, ir_pipe),
        daemon=True
    )

    imu_thread_obj.start()
    camera_thread_obj.start()

    frame_id = 0
    trajectory: List[np.ndarray] = []
    trajectory_slam: List[np.ndarray] = []
    loop_closure_poses: List[np.ndarray] = []

    try:
        while True:
            # Get the output from the queue with timeout
            try:
                timestamp, odom_pose, slam_pose, images = q.get(timeout=1.0)
            except queue.Empty:
                continue

            if odom_pose is None:
                continue

            frame_id += 1
            slam_initial_pose = vslam.Pose(translation=[0, 0, 0], rotation=[0, 0, 0, 1])
            current_pose = combine_poses(slam_initial_pose, odom_pose)
            trajectory.append(current_pose.translation)
            trajectory_slam.append(slam_pose.translation)   

            gravity = None
            if cfg.odometry_mode == vslam.Tracker.OdometryMode.Inertial:
                # Gravity estimation requires sufficient keyframes with motion
                gravity = tracker.get_last_gravity()

            raw_final_landmarks = list(tracker.get_final_landmarks().values())
            final_landmarks = transform_landmarks(raw_final_landmarks, slam_initial_pose)

            current_lc_poses = tracker.get_loop_closure_poses()
            if (current_lc_poses and 
                (not loop_closure_poses or 
                not np_array_equal(current_lc_poses[-1].pose.translation, loop_closure_poses[-1]))):
                loop_closure_poses.append(current_lc_poses[-1].pose.translation)    

            # Visualize results for left camera
            visualizer.visualize_frame(
                frame_id=frame_id,
                images=images,
                # pose=odom_pose,
                pose=current_pose,
                observations_main_cam=[tracker.get_last_observations(0), tracker.get_last_observations(1)],
                trajectory=trajectory,
                trajectory_slam=trajectory_slam,
                loop_closure_poses=loop_closure_poses,
                timestamp=timestamp,
                current_landmarks=tracker.get_last_landmarks(),
                final_landmarks=final_landmarks,
                gravity=gravity
            )

    except KeyboardInterrupt:
        print("Stopping VIO tracking...")
    finally:
        motion_pipe.stop()
        ir_pipe.stop()


if __name__ == "__main__":
    main()
