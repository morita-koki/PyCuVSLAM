# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

PyCuVSLAM is the official Python wrapper for NVIDIA cuVSLAM library, providing CUDA-accelerated Visual Tracking Camera modes and Simultaneous Localization and Mapping (SLAM) capabilities. The codebase supports both x86_64 (desktop/laptop) and aarch64 (Jetson) platforms.

## Installation and Setup Commands

### Install PyCuVSLAM package
```bash
# For x86_64 systems
pip install -e bin/x86_64

# For Jetson (aarch64) systems  
pip install -e bin/aarch64
```

### Install example dependencies
```bash
pip install -r examples/requirements.txt
```

### Activate virtual environment (if using venv)
```bash
source .venv/bin/activate
```

## Architecture and Structure

### Core Components

- **`bin/`**: Contains platform-specific Python packages
  - `bin/x86_64/`: x86_64 binaries and setup.py for desktop/laptop systems
  - `bin/aarch64/`: ARM64 binaries and setup.py for Jetson platforms
  - Both contain `cuvslam/` module with tracker.py as the main interface

- **`examples/`**: Comprehensive examples for different camera setups and datasets
  - `euroc/`: EuRoC dataset examples (monocular VO, stereo-inertial odometry)
  - `kitti/`: KITTI dataset examples (stereo VO, SLAM mapping/localization)
  - `realsense/`: Intel RealSense camera examples (stereo, mono-depth, VIO, multi-camera)
  - `zed/`: ZED camera examples
  - `oak-d/`: OAK-D camera examples
  - `tum/`: TUM dataset examples
  - `multicamera_edex/`: Multi-camera examples

- **`utils/`**: Utility modules for data processing
  - `utils/image/decompress.py`: ImageDecompressor for handling Bayer patterns and format conversion
  - `utils/ros/`: ROS-related utilities (rosbag processing, image extraction)

- **`docs/`**: API documentation (HTML files)

### Key Dependencies

- numpy==2.2.4
- pillow==11.1.0  
- pyyaml==6.0.2
- rerun-sdk==0.22.1 (for visualization)
- scipy==1.14.1

## Development Patterns

### Camera Tracking Modes

The library supports multiple tracking modes:
- **Monocular Visual Odometry**: Single camera tracking
- **Stereo Visual Odometry**: Dual camera tracking  
- **Monocular-Depth**: Single camera with depth information
- **Stereo Visual-Inertial**: Stereo cameras with IMU data
- **Multi-Camera**: Multiple synchronized cameras
- **SLAM**: Full mapping, localization, and map persistence

### Example Structure Pattern

Most examples follow this pattern:
1. Dataset path setup
2. Camera configuration/calibration
3. cuVSLAM tracker initialization
4. Rerun visualization setup
5. Frame processing loop with tracking and visualization

### Image Processing

- Use `ImageDecompressor` from `utils/image/decompress.py` for Bayer pattern conversion
- Supports formats: 'bayer_rggb8', 'bayer_grbg8', 'bayer_gbrg8', 'bayer_bggr8', 'rgb8'
- Handles grayscale to RGB conversion automatically

## Platform Considerations

### System Requirements
- **x86_64**: Ubuntu 22.04/24.04, NVIDIA GPU with CUDA 12.6, Python 3.10
- **aarch64**: Jetson with Jetpack 6.1/6.2, Python 3.10, CUDA 12.6

### Installation Methods
- Native install (Ubuntu 22.04 only)
- Virtual environment (venv)
- Conda environment 
- Docker containers (with RealSense integration)

## Performance Optimization

Key factors affecting performance:
- Hardware synchronization for multi-camera setups
- Proper camera calibration (intrinsic/extrinsic parameters)
- Appropriate frame rate (typically 30 FPS for human-speed motion)
- VGA resolution or higher recommended
- Minimize motion blur with proper exposure settings
- Use image masking for static/dynamic objects when needed

## Common Issues

- **Git LFS required**: Binary files need `git-lfs install && git lfs pull`
- **Python version**: Only Python 3.10 currently supported
- **CUDA compatibility**: Requires CUDA 12.6
- **Conda setup**: Set `LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH` when using conda