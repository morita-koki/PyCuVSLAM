#!/usr/bin/env python3
"""
ROSbag file information display script
Analyze bag files without installing ROS using rosbags library
"""

import argparse
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

try:
    from rosbags.highlevel import AnyReader
    from rosbags.typesys import Stores, get_typestore
except ImportError:
    print("Error: rosbags library is not installed.")
    print("Install command: pip install rosbags")
    sys.exit(1)


def detect_bag_version(bag_path: Path) -> str:
    """Detect bag file version"""
    if bag_path.is_file() and bag_path.suffix == '.bag':
        return 'ROS1'
    elif bag_path.is_dir() and (bag_path / 'metadata.yaml').exists():
        return 'ROS2'
    else:
        raise ValueError(f"Invalid bag file format: {bag_path}")


def get_compression_info(msg_type: str, msg_data) -> str:
    """Get compression image information"""
    compression_info = "None"
    
    # Check for compressed image
    if 'CompressedImage' in msg_type:
        if hasattr(msg_data, 'format'):
            compression_info = f"Compressed: {msg_data.format}"
        else:
            compression_info = "Compressed (format unknown)"
    
    # Check for Bayer format
    elif 'Image' in msg_type:
        if hasattr(msg_data, 'encoding'):
            encoding = msg_data.encoding
            if 'bayer' in encoding.lower():
                compression_info = f"Bayer: {encoding}"
            elif 'rgb' in encoding.lower() or 'bgr' in encoding.lower():
                compression_info = f"Uncompressed: {encoding}"
            else:
                compression_info = f"Encoding: {encoding}"
    
    return compression_info


def analyze_bag(bag_path: Path) -> Dict:
    """Analyze bag file (unified for ROS1/ROS2)"""
    topic_info = defaultdict(lambda: {
        'count': 0,
        'msg_type': '',
        'first_time': None,
        'last_time': None,
        'compression': 'None'
    })
    
    version = detect_bag_version(bag_path)
    print(f"Analyzing bag file: {bag_path} ({version})")
    
    try:
        # Prepare type store for old ROS2 bag files
        typestore = None
        if version == 'ROS2':
            typestore = get_typestore(Stores.ROS2_FOXY)
        
        # Use AnyReader for unified ROS1/ROS2 processing
        if typestore:
            reader = AnyReader([bag_path], default_typestore=typestore)
        else:
            reader = AnyReader([bag_path])
            
        with reader:
            # Initialize bag information
            bag_info = {
                'version': version,
                'duration': 0,
                'message_count': 0,
                'topics': {}
            }
            
            # Get topic information from connections
            for connection in reader.connections:
                topic = connection.topic
                msg_type = connection.msgtype
                topic_info[topic]['msg_type'] = msg_type
            
            # Read messages and collect statistics
            first_msg_processed = {}
            for connection, timestamp, rawdata in reader.messages():
                topic = connection.topic
                info = topic_info[topic]
                info['count'] += 1
                
                # Record timestamps (integer in nanoseconds)
                if info['first_time'] is None:
                    info['first_time'] = timestamp
                info['last_time'] = timestamp
                
                # Get compression info (only for first message of each topic)
                if topic not in first_msg_processed and info['compression'] == 'None':
                    try:
                        msg = reader.deserialize(rawdata, connection.msgtype)
                        info['compression'] = get_compression_info(connection.msgtype, msg)
                        first_msg_processed[topic] = True
                    except Exception as e:
                        # Skip if deserialization fails
                        print(f"Warning: Failed to analyze message for {topic}: {e}")
                        first_msg_processed[topic] = True
            
            # Calculate statistics
            for topic, info in topic_info.items():
                if info['count'] > 1 and info['first_time'] and info['last_time']:
                    # Convert nanoseconds to seconds
                    duration = (info['last_time'] - info['first_time']) * 1e-9
                    frequency = (info['count'] - 1) / duration if duration > 0 else 0
                else:
                    frequency = 0
                
                bag_info['topics'][topic] = {
                    'msg_type': info['msg_type'],
                    'count': info['count'],
                    'frequency': frequency,
                    'compression': info['compression']
                }
            
            bag_info['message_count'] = sum(info['count'] for info in topic_info.values())
            
            # Calculate total duration
            all_timestamps = []
            for info in topic_info.values():
                if info['first_time'] is not None:
                    all_timestamps.append(info['first_time'])
                if info['last_time'] is not None:
                    all_timestamps.append(info['last_time'])
            
            if all_timestamps:
                bag_info['duration'] = (max(all_timestamps) - min(all_timestamps)) * 1e-9
            
            return bag_info
            
    except Exception as e:
        print(f"Error: Failed to read bag file: {e}")
        return None


def print_bag_info(bag_info: Dict):
    """Display bag information in organized format"""
    if not bag_info:
        return
    
    print("\n" + "="*100)
    print(f"ROSbag File Information")
    print("="*100)
    print(f"ROS Version: {bag_info['version']}")
    print(f"Total Messages: {bag_info['message_count']:,}")
    print(f"Topic Count: {len(bag_info['topics'])}")
    if bag_info['duration'] > 0:
        print(f"Duration: {bag_info['duration']:.2f}s ({bag_info['duration']/60:.1f}min)")
    
    print("\n" + "-"*100)
    print("Topic Details:")
    print("-"*100)
    
    # Header
    print(f"{'Topic Name':<35} {'Message Type':<35} {'Msg Count':<12} {'Frequency(Hz)'}")
    print("-"*100)
    
    # Display topic information (sorted by message count in descending order)
    sorted_topics = sorted(bag_info['topics'].items(), key=lambda x: x[1]['count'], reverse=True)
    
    for topic, info in sorted_topics:
        frequency_str = f"{info['frequency']:.2f}" if info['frequency'] > 0 else "N/A"
        msg_type_short = info['msg_type'].split('/')[-1] if '/' in info['msg_type'] else info['msg_type']
        
        # Combine message type with compression/encoding info
        if info['compression'] != 'None':
            # Extract the specific encoding/compression format
            if 'Compressed: ' in info['compression']:
                format_info = info['compression'].replace('Compressed: ', '').split(';')[0]
                msg_type_with_info = f"{msg_type_short}({format_info})"
            elif 'Bayer: ' in info['compression']:
                format_info = info['compression'].replace('Bayer: ', '')
                msg_type_with_info = f"{msg_type_short}({format_info})"
            elif 'Uncompressed: ' in info['compression']:
                format_info = info['compression'].replace('Uncompressed: ', '')
                msg_type_with_info = f"{msg_type_short}({format_info})"
            elif 'Encoding: ' in info['compression']:
                format_info = info['compression'].replace('Encoding: ', '')
                msg_type_with_info = f"{msg_type_short}({format_info})"
            else:
                msg_type_with_info = f"{msg_type_short}({info['compression']})"
        else:
            msg_type_with_info = msg_type_short
        
        # Handle long string truncation
        topic_display = topic if len(topic) <= 34 else topic[:31] + "..."
        msg_type_display = msg_type_with_info if len(msg_type_with_info) <= 34 else msg_type_with_info[:31] + "..."
        
        print(f"{topic_display:<35} {msg_type_display:<35} {info['count']:<12,} {frequency_str}")
    
    print("-"*100)
    print(f"Total: {len(bag_info['topics'])} topics, {bag_info['message_count']:,} messages")


def main():
    parser = argparse.ArgumentParser(
        description="Display ROSbag file information",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python rosbag_info.py sample.bag
  python rosbag_info.py /path/to/ros2_bag_directory/
  
Note: rosbags library is required
  pip install rosbags
        """
    )
    parser.add_argument("bagfile", help="Path to ROSbag file or directory")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show detailed information")
    
    args = parser.parse_args()
    
    bag_path = Path(args.bagfile)
    
    if not bag_path.exists():
        print(f"Error: File or directory does not exist: {bag_path}")
        sys.exit(1)
    
    try:
        # Analyze bag
        bag_info = analyze_bag(bag_path)
        
        # Display results
        print_bag_info(bag_info)
        
    except Exception as e:
        print(f"Error: Failed to analyze bag file: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()