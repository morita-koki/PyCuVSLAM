#!/usr/bin/env python3
"""
Save Intel RealSense D435i camera calibration parameters to YAML
- Intrinsic parameters for IR-left, IR-right, and RGB cameras
- Extrinsic parameters of IR-right relative to IR-left
Generated with Claude Code
"""

import argparse
from pathlib import Path

import numpy as np
import pyrealsense2 as rs
import yaml


def get_camera_intrinsics(profile) -> dict:
    """Extract intrinsic parameters from camera profile"""
    intrinsics = profile.get_intrinsics()
    
    return {
        'width': intrinsics.width,
        'height': intrinsics.height,
        'fx': intrinsics.fx,
        'fy': intrinsics.fy,
        'cx': intrinsics.ppx,
        'cy': intrinsics.ppy,
        'distortion_model': str(intrinsics.model),
        'distortion_coefficients': list(intrinsics.coeffs)
    }


def get_camera_extrinsics(from_profile, to_profile) -> dict:
    """Extract extrinsic parameters between two camera profiles"""
    extrinsics = from_profile.get_extrinsics_to(to_profile)
    
    # Convert rotation matrix to 3x3 format
    rotation_matrix = np.array(extrinsics.rotation).reshape(3, 3)
    
    # Translation vector
    translation_vector = np.array(extrinsics.translation)
    
    return {
        'rotation_matrix': rotation_matrix.tolist(),
        'translation_vector': translation_vector.tolist(),
        'rotation_matrix_flat': list(extrinsics.rotation),
        'translation_vector_flat': list(extrinsics.translation)
    }


def save_d435i_calibration(workdir: str = "./", output_file: str = None):
    """Extract and save D435i calibration parameters"""
    
    # Initialize RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    
    # Enable streams
    config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)  # IR Left
    config.enable_stream(rs.stream.infrared, 2, 640, 480, rs.format.y8, 30)  # IR Right
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)      # RGB
    
    try:
        # Start pipeline
        profile = pipeline.start(config)
        
        # Get stream profiles
        ir_left_profile = profile.get_stream(rs.stream.infrared, 1).as_video_stream_profile()
        ir_right_profile = profile.get_stream(rs.stream.infrared, 2).as_video_stream_profile()
        color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        
        # Extract intrinsic parameters
        ir_left_intrinsics = get_camera_intrinsics(ir_left_profile)
        ir_right_intrinsics = get_camera_intrinsics(ir_right_profile)
        rgb_intrinsics = get_camera_intrinsics(color_profile)
        
        # Extract extrinsic parameters (IR-right relative to IR-left)
        ir_right_to_ir_left_extrinsics = get_camera_extrinsics(ir_left_profile, ir_right_profile)
        
        # Also get RGB relative to IR-left for completeness
        rgb_to_ir_left_extrinsics = get_camera_extrinsics(ir_left_profile, color_profile)
        
        # Convert intrinsics to the required format
        def format_intrinsic(intrinsics):
            return [intrinsics['fx'], 0.0, intrinsics['cx'],
                   0.0, intrinsics['fy'], intrinsics['cy'],
                   0.0, 0.0, 1.0]
        
        def format_extrinsic_4x4(rotation_matrix, translation_vector):
            """Convert rotation matrix and translation to 4x4 homogeneous matrix as flat list"""
            extrinsic_4x4 = []
            for i in range(3):
                for j in range(3):
                    extrinsic_4x4.append(rotation_matrix[i][j])
                extrinsic_4x4.append(translation_vector[i])
            extrinsic_4x4.extend([0, 0, 0, 1])
            return extrinsic_4x4
        
        # Identity extrinsic for IR-left (reference camera)
        identity_extrinsic = [1.0, 0.0, 0.0, 0.0,
                             0.0, 1.0, 0.0, 0.0, 
                             0.0, 0.0, 1.0, 0.0,
                             0, 0, 0, 1]
        
        # IR-right extrinsic relative to IR-left
        ir_right_extrinsic = format_extrinsic_4x4(
            ir_right_to_ir_left_extrinsics['rotation_matrix'],
            ir_right_to_ir_left_extrinsics['translation_vector']
        )
        
        # RGB extrinsic relative to IR-left
        rgb_extrinsic = format_extrinsic_4x4(
            rgb_to_ir_left_extrinsics['rotation_matrix'],
            rgb_to_ir_left_extrinsics['translation_vector']
        )
        
        # Set default output file if not provided
        if output_file is None:
            output_file = Path(workdir) / "settings.yaml"
        
        # Prepare calibration data in the requested format
        calibration_data = {
            'workdir': str(Path(workdir).absolute()),
            'cameras': [
                {
                    'name': 'ir-left',
                    'extrinsic': identity_extrinsic,
                    'intrinsic': format_intrinsic(ir_left_intrinsics),
                    'distortion': ir_left_intrinsics['distortion_coefficients'],
                    'resolution': [ir_left_intrinsics['width'], ir_left_intrinsics['height']]
                },
                {
                    'name': 'ir-right', 
                    'extrinsic': ir_right_extrinsic,
                    'intrinsic': format_intrinsic(ir_right_intrinsics),
                    'distortion': ir_right_intrinsics['distortion_coefficients'],
                    'resolution': [ir_right_intrinsics['width'], ir_right_intrinsics['height']]
                },
                {
                    'name': 'rgb',
                    'extrinsic': rgb_extrinsic,
                    'intrinsic': format_intrinsic(rgb_intrinsics),
                    'distortion': rgb_intrinsics['distortion_coefficients'],
                    'resolution': [rgb_intrinsics['width'], rgb_intrinsics['height']]
                }
            ]
        }
        
        # Custom YAML representer for lists
        def represent_list(dumper, data):
            # Check if all elements are numbers (int or float)
            if data and all(isinstance(x, (int, float)) for x in data):
                return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)
            else:
                return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=False)
        
        yaml.add_representer(list, represent_list)
        
        # Save to YAML file
        output_path = Path(output_file)
        with open(output_path, 'w') as f:
            yaml.dump(calibration_data, f, default_flow_style=False, sort_keys=False, indent=2, width=float('inf'))
        
        print(f"Calibration parameters saved to: {output_path.absolute()}")
        print(f"Device Serial Number: {profile.get_device().get_info(rs.camera_info.serial_number)}")
        print(f"Stereo Baseline: {abs(ir_right_to_ir_left_extrinsics['translation_vector'][0]) * 1000:.2f} mm")
        
        # Print summary
        print("\nIntrinsic Parameters Summary:")
        print(f"IR-Left:  fx={ir_left_intrinsics['fx']:.2f}, fy={ir_left_intrinsics['fy']:.2f}, cx={ir_left_intrinsics['cx']:.2f}, cy={ir_left_intrinsics['cy']:.2f}")
        print(f"IR-Right: fx={ir_right_intrinsics['fx']:.2f}, fy={ir_right_intrinsics['fy']:.2f}, cx={ir_right_intrinsics['cx']:.2f}, cy={ir_right_intrinsics['cy']:.2f}")
        print(f"RGB:      fx={rgb_intrinsics['fx']:.2f}, fy={rgb_intrinsics['fy']:.2f}, cx={rgb_intrinsics['cx']:.2f}, cy={rgb_intrinsics['cy']:.2f}")
        
        return calibration_data
        
    except Exception as e:
        print(f"Error: {e}")
        return None
        
    finally:
        pipeline.stop()


def main():
    parser = argparse.ArgumentParser(description='Save RealSense D435i calibration parameters to YAML')
    parser.add_argument('--workdir', '-w', type=str, default='./',
                        help='Working directory (default: ./)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output YAML file path (default: workdir/settings.yaml)')
    
    args = parser.parse_args()
    
    print("Extracting RealSense D435i calibration parameters...")
    print(f"Working directory: {args.workdir}")
    calibration_data = save_d435i_calibration(args.workdir, args.output)
    
    if calibration_data:
        print(" Calibration extraction completed successfully")
        return 0
    else:
        print(" Failed to extract calibration parameters")
        return 1


if __name__ == "__main__":
    exit(main())