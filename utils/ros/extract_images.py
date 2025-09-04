#!/usr/bin/env python3

import os
import sys
import yaml
import cv2
import numpy as np
import argparse
from pathlib import Path

def decompress_compressed_image(compressed_msg, debug=False):
    """
    Decompress CompressedImage message using OpenCV only
    
    Args:
        compressed_msg: CompressedImage message
        debug: Print debug information
    
    Returns:
        numpy.ndarray: OpenCV image (BGR format)
    """
    try:
        # Convert message data to numpy array
        np_arr = np.frombuffer(compressed_msg.data, np.uint8)
        
        # Decode image using OpenCV
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if cv_image is None:
            # Try with unchanged flag if color decode fails
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
            
        if cv_image is None:
            raise Exception("Failed to decode compressed image")
            
        if debug:
            print(f"    Decompressed image shape: {cv_image.shape}")
            if hasattr(compressed_msg, 'format'):
                print(f"    Compression format: {compressed_msg.format}")
            
        # Handle different channel configurations
        if len(cv_image.shape) == 2:
            # Grayscale image, convert to 3-channel BGR
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2BGR)
            if debug:
                print(f"    Converted grayscale to BGR")
        elif len(cv_image.shape) == 3:
            if cv_image.shape[2] == 4:
                # RGBA/BGRA image, convert to BGR
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGRA2BGR)
                if debug:
                    print(f"    Converted BGRA to BGR")
            elif cv_image.shape[2] == 3:
                # Already BGR or RGB - OpenCV decode usually gives BGR
                if debug:
                    print(f"    Image is 3-channel (assumed BGR)")
        
        # Check for Bayer pattern in format string
        if hasattr(compressed_msg, 'format') and 'bayer' in compressed_msg.format.lower():
            if len(cv_image.shape) == 2 or (len(cv_image.shape) == 3 and cv_image.shape[2] == 1):
                # Apply Bayer demosaicing
                format_lower = compressed_msg.format.lower()
                if 'rggb' in format_lower:
                    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BAYER_RG2BGR)
                elif 'grbg' in format_lower:
                    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BAYER_GR2BGR)
                elif 'gbrg' in format_lower:
                    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BAYER_GB2BGR)
                elif 'bggr' in format_lower:
                    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BAYER_BG2BGR)
                else:
                    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BAYER_RG2BGR)  # Default
                if debug:
                    print(f"    Applied Bayer demosaicing")
        
        return cv_image
        
    except Exception as e:
        raise Exception(f"Compressed image decompression failed: {e}")

def convert_ros_image(image_msg, debug=False):
    """
    Convert ROS Image message to OpenCV image using only OpenCV and numpy
    
    Args:
        image_msg: sensor_msgs/Image message
        debug: Print debug information
        
    Returns:
        numpy.ndarray: OpenCV image (BGR format)
    """
    try:
        # Get image properties
        height = image_msg.height
        width = image_msg.width
        encoding = image_msg.encoding
        step = image_msg.step
        data = image_msg.data
        
        if debug:
            print(f"    Image properties: {width}x{height}, encoding: {encoding}, step: {step}")
        
        # Convert data to numpy array
        dtype = np.uint8
        if '16' in encoding:
            dtype = np.uint16
        elif '32' in encoding:
            dtype = np.uint32
            
        # Create numpy array from message data
        np_arr = np.frombuffer(data, dtype=dtype)
        
        # Determine number of channels
        if encoding in ['mono8', 'mono16']:
            channels = 1
        elif encoding in ['rgb8', 'bgr8', 'rgb16', 'bgr16']:
            channels = 3
        elif encoding in ['rgba8', 'bgra8', 'rgba16', 'bgra16']:
            channels = 4
        elif 'bayer' in encoding.lower():
            channels = 1  # Bayer is single channel before demosaicing
        else:
            # Try to infer from data size
            expected_size = height * width
            if len(np_arr) == expected_size:
                channels = 1
            elif len(np_arr) == expected_size * 3:
                channels = 3
            elif len(np_arr) == expected_size * 4:
                channels = 4
            else:
                channels = 3  # Default assumption
                
        # Reshape array to image dimensions
        if channels == 1:
            cv_image = np_arr.reshape((height, width))
        else:
            cv_image = np_arr.reshape((height, width, channels))
            
        if debug:
            print(f"    Reshaped to: {cv_image.shape}, channels: {channels}")
        
        # Handle different encodings
        if encoding == 'rgb8':
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
        elif encoding == 'rgba8':
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGBA2BGR)
        elif encoding == 'bgra8':
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGRA2BGR)
        elif encoding in ['mono8', 'mono16']:
            # Convert grayscale to BGR
            if len(cv_image.shape) == 2:
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2BGR)
        elif 'bayer' in encoding.lower():
            # Handle Bayer patterns
            encoding_lower = encoding.lower()
            if 'rggb' in encoding_lower:
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BAYER_RG2BGR)
            elif 'grbg' in encoding_lower:
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BAYER_GR2BGR)
            elif 'gbrg' in encoding_lower:
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BAYER_GB2BGR)
            elif 'bggr' in encoding_lower:
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BAYER_BG2BGR)
            else:
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BAYER_RG2BGR)  # Default
            if debug:
                print(f"    Applied Bayer demosaicing for {encoding}")
        # bgr8 and bgr16 are already in correct format
        
        # Handle 16-bit images
        if dtype == np.uint16:
            # Convert to 8-bit for JPEG output
            cv_image = (cv_image / 256).astype(np.uint8)
            if debug:
                print(f"    Converted 16-bit to 8-bit")
                
        return cv_image
        
    except Exception as e:
        raise Exception(f"ROS Image conversion failed: {e}")

def load_config(config_path):
    """
    Load YAML configuration file
    
    Args:
        config_path: Path to the YAML config file
        
    Returns:
        dict: Parsed configuration
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
            return config
    except Exception as e:
        raise Exception(f"Failed to load config file {config_path}: {e}")

def extract_images_from_bag(config, output_dir=None, debug=False, max_images=None):
    """
    Extract images from ROS bag file based on configuration using AnyReader from rosbags library
    
    Args:
        config: Configuration dictionary
        output_dir: Output directory (overrides workdir from config)
        debug: Enable debug output
        max_images: Maximum number of images to extract (None for all)
        
    Returns:
        bool: Success status
    """
    # Get bag file path
    bag_file = config.get('bag')
    if not bag_file:
        print("Error: 'bag' field not found in configuration")
        return False
    
    # Check if bag file exists
    if not os.path.exists(bag_file):
        print(f"Error: Bag file '{bag_file}' not found")
        return False
    
    # Get working directory
    if output_dir is None:
        output_dir = config.get('workdir', '.')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get cameras configuration
    cameras = config.get('cameras', [])
    if not cameras:
        print("Error: No cameras found in configuration")
        return False
    
    try:
        # Import rosbags library
        from rosbags.highlevel import AnyReader
        from rosbags.typesys import Stores, get_typestore
        
        print("Using AnyReader from rosbags library")
        
        # Create type store for message definitions
        typestore = get_typestore(Stores.LATEST)
        
        # Convert bag file path to Path object
        bag_path = Path(bag_file)
        
        # Open bag file using AnyReader
        with AnyReader([bag_path], default_typestore=typestore) as reader:
            # Get available topics and types
            available_topics = {conn.topic: conn.msgtype for conn in reader.connections}
            print(f"Available topics in bag: {list(available_topics.keys())}")
            
            # Create connections filter for cameras
            camera_connections = []
            camera_map = {}  # connection -> camera_name mapping
            
            for camera in cameras:
                camera_name = camera.get('name', 'unknown')
                topic = camera.get('topic')
                
                if not topic:
                    print(f"Warning: No topic specified for camera '{camera_name}', skipping")
                    continue
                
                if topic not in available_topics:
                    print(f"Warning: Topic '{topic}' not found in bag file for camera '{camera_name}', skipping")
                    continue
                
                # Find connections for this topic
                connections = [conn for conn in reader.connections if conn.topic == topic]
                if connections:
                    camera_connections.extend(connections)
                    for conn in connections:
                        camera_map[conn] = camera_name
                    
                    topic_type = available_topics[topic]
                    print(f"Found camera '{camera_name}' (topic: {topic}, type: {topic_type})")
            
            if not camera_connections:
                print("Error: No valid camera connections found")
                return False
            
            # Initialize counters for each camera
            frame_counts = {camera_name: 0 for camera_name in set(camera_map.values())}
            error_counts = {camera_name: 0 for camera_name in set(camera_map.values())}
            processed_cameras = set(camera_map.values())
            
            print(f"Starting to process {len(camera_connections)} connections...")
            
            # Read messages using connections parameter
            for connection, timestamp, rawdata in reader.messages(connections=camera_connections):
                camera_name = camera_map[connection]
                topic_type = connection.msgtype
                
                # Skip if this camera has reached max_images
                if max_images and frame_counts[camera_name] >= max_images:
                    continue
                
                try:
                    # Deserialize message
                    msg = reader.deserialize(rawdata, connection.msgtype)
                    
                    # Use timestamp as nanoseconds
                    timestamp_ns = timestamp
                    
                    # Check if message type is supported
                    if 'CompressedImage' not in topic_type and 'Image' not in topic_type:
                        if error_counts[camera_name] == 0:
                            print(f"Warning: Unsupported message type '{topic_type}' for camera '{camera_name}', skipping")
                        error_counts[camera_name] += 1
                        continue
                    
                    # is_compressed = 'CompressedImage' in topic_type
                    
                    # Process image based on message type
                    # if is_compressed:
                    #     if debug and frame_counts[camera_name] < 3:
                    #         print(f"  Processing compressed frame {frame_counts[camera_name] + 1} for camera '{camera_name}'")
                        
                    #     cv_image = decompress_compressed_image(msg, debug=(debug and frame_counts[camera_name] < 3))
                    # else:
                    # Regular Image message
                    # if debug and frame_counts[camera_name] < 3:
                    #     print(f"  Processing regular frame {frame_counts[camera_name] + 1} for camera '{camera_name}'")
                        
                    # cv_image = convert_ros_image(msg, debug=(debug and frame_counts[camera_name] < 3))
                    
                    np_arr = np.frombuffer(msg.data, np.uint8)
                    cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                    
                    # Generate output filename
                    filename = f"{output_dir}/camera/{camera_name}/{timestamp_ns}.jpg"
                    os.makedirs(os.path.dirname(filename), exist_ok=True)

                    # Save image
                    success = cv2.imwrite(filename, cv_image)
                    
                    if success:
                        frame_counts[camera_name] += 1
                        if frame_counts[camera_name] % 100 == 0:
                            print(f"  Extracted {frame_counts[camera_name]} images for camera '{camera_name}'")
                        
                        # Check if we've reached the maximum number of images
                        if max_images and frame_counts[camera_name] >= max_images:
                            print(f"  Reached maximum image limit ({max_images}) for camera '{camera_name}'")
                            processed_cameras.discard(camera_name)
                            if not processed_cameras:  # No more cameras to process
                                break
                    else:
                        print(f"  Failed to save image: {filename}")
                        error_counts[camera_name] += 1
                        
                except Exception as e:
                    error_counts[camera_name] += 1
                    if error_counts[camera_name] <= 3:
                        print(f"  Error processing frame {frame_counts[camera_name]} for camera '{camera_name}': {e}")
                    elif error_counts[camera_name] == 4:
                        print(f"  Too many errors, suppressing further error messages for camera '{camera_name}'")
                    continue
            
            # Print final statistics
            total_success = 0
            for camera_name in frame_counts:
                frame_count = frame_counts[camera_name]
                error_count = error_counts[camera_name]
                
                if frame_count > 0:
                    print(f"  Successfully extracted {frame_count} images for camera '{camera_name}'")
                    if error_count > 0:
                        print(f"    ({error_count} frames failed to process)")
                    total_success += frame_count
                else:
                    print(f"  No valid images found for camera '{camera_name}'")
            
            return total_success > 0
        
    except ImportError as e:
        print(f"Error: rosbags library not found. Please install it with: pip install rosbags")
        print(f"Import error details: {e}")
        return False
    except Exception as e:
        print(f"Error processing bag file: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Extract images from ROS bag file using AnyReader from rosbags library (pure Python, no ROS dependency)')
    parser.add_argument('config', help='Path to YAML configuration file')
    parser.add_argument('-o', '--output-dir', 
                       help='Output directory (overrides workdir from config)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug output')
    parser.add_argument('--max-images', type=int,
                       help='Maximum number of images to extract per camera (default: extract all)')
    
    args = parser.parse_args()
    
    print("ROS Bag Image Extractor (using AnyReader from rosbags library)")
    print("=" * 65)
    
    try:
        # Load configuration
        config = load_config(args.config)
        print(f"Loaded configuration from: {args.config}")
        
        # Print configuration summary
        print(f"Bag file: {config.get('bag', 'Not specified')}")
        print(f"Work directory: {config.get('workdir', 'Not specified')}")
        if args.output_dir:
            print(f"Output directory (override): {args.output_dir}")
        
        cameras = config.get('cameras', [])
        print(f"Number of cameras: {len(cameras)}")
        for i, camera in enumerate(cameras):
            print(f"  {i+1}. {camera.get('name', 'unknown')} -> {camera.get('topic', 'no topic')}")
        
        # Extract images
        success = extract_images_from_bag(
            config, 
            output_dir=args.output_dir, 
            debug=args.debug,
            max_images=args.max_images
        )
        
        if success:
            print("\nImage extraction completed successfully!")
            sys.exit(0)
        else:
            print("\nImage extraction failed!")
            sys.exit(1)
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()