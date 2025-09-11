#!/usr/bin/env python3
"""
Record Intel RealSense D435i stereo IR and RGB camera images
Generated with Claude Code
"""

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
import rerun as rr


def create_directories(output_dir: str) -> dict:
    """Create output directories for cameras and return paths"""
    base_path = Path(output_dir)
    
    # Create camera directories
    camera_dirs = {
        'ir-left': base_path / 'camera' / 'ir-left',
        'ir-right': base_path / 'camera' / 'ir-right', 
        'rgb': base_path / 'camera' / 'rgb'
    }
    
    for camera_dir in camera_dirs.values():
        camera_dir.mkdir(parents=True, exist_ok=True)
        
    return camera_dirs


def initialize_camera(width: int = 640, height: int = 480, fps: int = 30) -> tuple:
    """Initialize RealSense D435i camera with stereo IR and RGB streams"""
    pipeline = rs.pipeline()
    config = rs.config()
    
    # Enable stereo infrared streams
    config.enable_stream(rs.stream.infrared, 1, width, height, rs.format.y8, fps)  # Left IR
    config.enable_stream(rs.stream.infrared, 2, width, height, rs.format.y8, fps)  # Right IR
    
    # Enable RGB stream
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    
    # Start pipeline
    profile = pipeline.start(config)
    
    # Warm up camera
    for _ in range(30):
        pipeline.wait_for_frames()
    
    return pipeline, profile


def record_frames(pipeline: rs.pipeline, camera_dirs: dict, output_dir: str, frequency: int = 30, enable_rerun: bool = True):
    """Record synchronized frames from all cameras"""
    frame_interval = 1.0 / frequency
    csv_path = Path(output_dir) / 'timestamp.csv'
    
    # Initialize Rerun if enabled
    if enable_rerun:
        rr.init("RealSense D435i Recording", spawn=True)
        rr.log("description", rr.TextDocument("Recording stereo IR and RGB from RealSense D435i", media_type=rr.MediaType.MARKDOWN))
    
    print(f"Recording at {frequency}Hz... Press Ctrl+C to stop")
    if enable_rerun:
        print("Rerun visualization enabled")
    
    frame_count = 0
    
    try:
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['frame_id', 'timestamp'])
            
            start_time = time.time()
            next_capture_time = start_time
            
            while True:
                current_time = time.time()
                
                # Wait until it's time for the next frame
                if current_time < next_capture_time:
                    time.sleep(0.001)  # Small sleep to avoid busy waiting
                    continue
                
                # Capture frames
                frames = pipeline.wait_for_frames()
                
                # Get timestamp (use frame timestamp for better synchronization)
                timestamp = frames.get_timestamp() / 1000.0  # Convert to seconds
                
                # Get individual frames
                left_ir_frame = frames.get_infrared_frame(1)
                right_ir_frame = frames.get_infrared_frame(2)
                color_frame = frames.get_color_frame()
                
                if not (left_ir_frame and right_ir_frame and color_frame):
                    continue
                
                # Convert to numpy arrays
                left_ir_image = np.asanyarray(left_ir_frame.get_data())
                right_ir_image = np.asanyarray(right_ir_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())
                
                # Save images with timestamp as filename
                timestamp_str = f"{timestamp:.6f}"
                
                # Save IR images (convert to 3-channel for JPEG)
                left_ir_rgb = cv2.cvtColor(left_ir_image, cv2.COLOR_GRAY2BGR)
                right_ir_rgb = cv2.cvtColor(right_ir_image, cv2.COLOR_GRAY2BGR)
                
                cv2.imwrite(str(camera_dirs['ir-left'] / f"{timestamp_str}.jpg"), left_ir_rgb)
                cv2.imwrite(str(camera_dirs['ir-right'] / f"{timestamp_str}.jpg"), right_ir_rgb)
                cv2.imwrite(str(camera_dirs['rgb'] / f"{timestamp_str}.jpg"), color_image)
                
                # Log to Rerun for visualization
                if enable_rerun:
                    rr.set_time_seconds("timestamp", timestamp)
                    
                    # Log stereo IR images
                    rr.log("cameras/ir-left", rr.Image(left_ir_image))
                    rr.log("cameras/ir-right", rr.Image(right_ir_image))
                    
                    # Log RGB image (convert BGR to RGB for proper display)
                    rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
                    rr.log("cameras/rgb", rr.Image(rgb_image))
                    
                    # Log stereo pair for comparison
                    # stereo_pair = np.hstack((left_ir_image, right_ir_image))
                    # rr.log("cameras/stereo_pair", rr.Image(stereo_pair))
                
                # Write timestamp to CSV
                writer.writerow([frame_count, timestamp_str])
                
                frame_count += 1
                next_capture_time += frame_interval
                
                if frame_count % 30 == 0:
                    print(f"Recorded {frame_count} frames...")
                    
    except KeyboardInterrupt:
        print(f"\nRecording stopped. Total frames: {frame_count}")
    except Exception as e:
        print(f"Error during recording: {e}")
    finally:
        pipeline.stop()


def main():
    parser = argparse.ArgumentParser(description='Record RealSense D435i stereo IR and RGB images')
    parser.add_argument('--output-dir', '-o', type=str, default='./recorded_data',
                        help='Output directory for recorded data (default: ./recorded_data)')
    parser.add_argument('--frequency', '-f', type=int, default=30,
                        help='Recording frequency in Hz (default: 30)')
    parser.add_argument('--width', type=int, default=640,
                        help='Image width (default: 640)')
    parser.add_argument('--height', type=int, default=480,
                        help='Image height (default: 480)')
    parser.add_argument('--no-rerun', action='store_true',
                        help='Disable Rerun visualization')
    
    args = parser.parse_args()
    
    # Create output directories
    camera_dirs = create_directories(args.output_dir)
    
    print(f"Output directory: {args.output_dir}")
    print(f"Recording frequency: {args.frequency} Hz")
    print(f"Resolution: {args.width}x{args.height}")
    
    # Initialize camera
    try:
        pipeline, _ = initialize_camera(args.width, args.height, args.frequency)
        print("Camera initialized successfully")
        
        # Record frames
        record_frames(pipeline, camera_dirs, args.output_dir, args.frequency, enable_rerun=not args.no_rerun)
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())