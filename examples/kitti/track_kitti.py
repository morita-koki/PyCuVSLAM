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
from numpy import loadtxt, asarray
from PIL import Image
import rerun as rr
import rerun.blueprint as rrb
import cuvslam

# Set up dataset path
sequence_path = os.path.join(
    os.path.dirname(__file__),
    "dataset/sequences/06"
)

sequence_path = '/media/morita/SSD_B0084/kitti/02'

# Generate pseudo-random colour from integer identifier for visualization
def color_from_id(identifier): 
    return [(identifier * 17) % 256, (identifier * 31) % 256, (identifier * 47) % 256]

# Setup rerun visualizer
rr.init('kitti', strict=True, spawn=True)  # launch re-run instance

# Setup rerun views
rr.send_blueprint(rrb.Blueprint(
    rrb.TimePanel(state="collapsed"),
    rrb.Vertical(
        row_shares=[0.6, 0.4],
        contents=[rrb.Spatial3DView(), rrb.Spatial2DView(origin='car/cam0')]
    )
))

# Setup coordinate basis for root, cuvslam uses right-hand system with X-right, Y-down, Z-forward
rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)

# Draw arrays in origin X-red, Y-green, Z-blue
rr.log("xyz", rr.Arrows3D(
    vectors=[[50, 0, 0], [0, 50, 0], [0, 0, 50]],
    colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
    labels=['[x]', '[y]', '[z]']
), static=True)

# Load KITTI dataset calibration and initilize cameras
intrinsics = loadtxt(
    os.path.join(sequence_path, 'calib.txt'),
    usecols=range(1, 13)
)[:4].reshape(4, 3, 4)

size = Image.open(os.path.join(sequence_path, 'image_2', '000001.png')).size

cameras = [cuvslam.Camera(), cuvslam.Camera()]
for i in [0, 1]:
    cameras[i].size = size
    cameras[i].principal = [intrinsics[i][0][2], intrinsics[i][1][2]]
    cameras[i].focal = [intrinsics[i].diagonal()[0], intrinsics[i].diagonal()[1]]
cameras[1].rig_from_camera.translation[0] = -intrinsics[1][0][3] / intrinsics[1][0][0]

# Initialize the cuvslam tracker
cfg = cuvslam.Tracker.OdometryConfig(
    async_sba=False,
    enable_final_landmarks_export=True,
    horizontal_stereo_camera=True
)
tracker = cuvslam.Tracker(cuvslam.Rig(cameras), cfg)

# Get timestamps from times.txt file
timestamps = [
    int(10 ** 9 * float(sec_str))
    for sec_str in open(os.path.join(sequence_path, 'times.txt')).readlines()
]

# Track each frames in the dataset sequence
trajectory = []
for frame in range(len(timestamps)):
    # Load grayscale pixels as array for left and right absolute image paths
    images = [
        asarray(Image.open(os.path.join(sequence_path, f'image_{cam+2}', f'{frame:0>6}.png')))
        for cam in [0, 1]
    ]

    # Do tracking
    odom_pose_estimate, _ = tracker.track(timestamps[frame], images)

    if odom_pose_estimate.world_from_rig is None:
        print(f"Warning: Failed to track frame {frame}")
        continue

    # Get current pose and observations for the main camera and gravity in rig frame
    odom_pose = odom_pose_estimate.world_from_rig.pose

    # Get visualization data
    observations = tracker.get_last_observations(0)  # get observation from left camera
    landmarks = tracker.get_last_landmarks()
    final_landmarks = tracker.get_final_landmarks()

    # Prepare visualization data
    observations_uv = [[o.u, o.v] for o in observations]
    observations_colors = [color_from_id(o.id) for o in observations]
    landmark_xyz = [l.coords for l in landmarks]
    landmarks_colors = [color_from_id(l.id) for l in landmarks]
    trajectory.append(odom_pose.translation)

    # Send results to rerun for visualization
    rr.set_time_sequence('frame', frame)
    rr.log('trajectory', rr.LineStrips3D(trajectory))
    rr.log('final_landmarks', rr.Points3D(list(final_landmarks.values()), radii=0.1))
    rr.log('car', rr.Transform3D(
        translation=odom_pose.translation,
        quaternion=odom_pose.rotation
    ))
    rr.log('car/body', rr.Boxes3D(centers=[0, 1.65 / 2, 0], sizes=[[1.6, 1.65, 2.71]]))
    rr.log('car/landmarks_center', rr.Points3D(
        landmark_xyz, radii=0.25, colors=landmarks_colors
    ))
    rr.log('car/landmarks_lines', rr.Arrows3D(
        vectors=landmark_xyz, radii=0.05, colors=landmarks_colors
    ))
    rr.log('car/cam0', rr.Pinhole(
        image_plane_distance=1.68,
        image_from_camera=intrinsics[0][:3, :3],
        width=size[0],
        height=size[1]
    ))
    rr.log('car/cam0/image', rr.Image(images[0]).compress(jpeg_quality=80))
    rr.log('car/cam0/observations', rr.Points2D(
        observations_uv, radii=5, colors=observations_colors
    ))
