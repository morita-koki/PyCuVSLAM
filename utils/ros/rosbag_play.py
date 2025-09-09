"""
Rosbag Image Topic Player with Rerun Visualization

This script plays back image topics from rosbag files using rerun for visualization.
Images are automatically downscaled to maximum 1024px in both width and height to reduce file size.
All image topics in the rosbag are displayed simultaneously with topic synchronization.
No ROS dependencies required - uses rosbags library for pure Python parsing.

🤖 Generated with Claude Code (https://claude.ai/code)
"""

import argparse
from rosbags.highlevel import AnyReader
import cv2
import numpy as np
import rerun as rr
import time
from typing import List, Any, Dict, Optional
from pathlib import Path
from collections import deque
import heapq
from tqdm import tqdm


class RosbagImagePlayer:
    def __init__(self, max_resolution: int = 1024):
        """
        Initialize the rosbag image player.
        
        Args:
            max_resolution: Maximum resolution for width or height (default: 1024)
        """
        self.max_resolution = max_resolution
        self.image_topics: List[str] = []
        
    def _downscale_image(self, image: np.ndarray) -> np.ndarray:
        """
        Downscale image to maximum resolution while maintaining aspect ratio.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Downscaled image
        """
        height, width = image.shape[:2]
        
        if max(height, width) <= self.max_resolution:
            return image
            
        if width > height:
            new_width = self.max_resolution
            new_height = int(height * self.max_resolution / width)
        else:
            new_height = self.max_resolution
            new_width = int(width * self.max_resolution / height)
            
        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    
    def _get_image_topics(self, bag_path: str) -> List[str]:
        """
        Extract all image topics from the rosbag.
        
        Args:
            bag_path: Path to the rosbag file or directory
            
        Returns:
            List of image topic names
        """
        topics = []
        
        with AnyReader([Path(bag_path)]) as reader:
            for connection in reader.connections:
                if connection.msgtype in ['sensor_msgs/msg/Image', 'sensor_msgs/msg/CompressedImage',
                                        'sensor_msgs/Image', 'sensor_msgs/CompressedImage']:
                    if connection.topic not in topics:
                        topics.append(connection.topic)
                        
        return topics
    
    def _decode_image_message(self, msg_data: Any, msg_type: str) -> np.ndarray:
        """
        Decode image message data to numpy array.
        
        Args:
            msg_data: Deserialized message data
            msg_type: Message type string
            
        Returns:
            OpenCV image as numpy array
        """
        if 'CompressedImage' in msg_type:
            # Handle CompressedImage
            compressed_data = msg_data.data
            np_arr = np.frombuffer(compressed_data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        else:
            # Handle regular Image message
            width = msg_data.width
            height = msg_data.height
            encoding = msg_data.encoding
            data = msg_data.data
            
            # Convert data to numpy array based on encoding
            if encoding == 'rgb8':
                cv_image = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 3))
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
            elif encoding == 'bgr8':
                cv_image = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 3))
            elif encoding == 'mono8':
                cv_image = np.frombuffer(data, dtype=np.uint8).reshape((height, width))
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2BGR)
            elif encoding == 'mono16':
                cv_image = np.frombuffer(data, dtype=np.uint16).reshape((height, width))
                cv_image = (cv_image / 256).astype(np.uint8)
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2BGR)
            else:
                # Try to handle other encodings as 8-bit
                try:
                    channels = len(data) // (width * height)
                    if channels == 3:
                        cv_image = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 3))
                    else:
                        cv_image = np.frombuffer(data, dtype=np.uint8).reshape((height, width))
                        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2BGR)
                except:
                    print(f"Unsupported encoding: {encoding}")
                    return None
                    
        return cv_image
    
    def _process_image_message(self, msg_data: Any, msg_type: str, topic: str, timestamp: float) -> None:
        """
        Process and display image message in rerun.
        
        Args:
            msg_data: Deserialized message data
            msg_type: Message type string
            topic: Topic name
            timestamp: Message timestamp
        """
        try:
            # Decode image message
            cv_image = self._decode_image_message(msg_data, msg_type)
            
            if cv_image is None:
                print(f"Failed to decode image from topic {topic}")
                return
                
            # Convert BGR to RGB for rerun
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            
            # Downscale image
            rgb_image = self._downscale_image(rgb_image)
            
            # Clean topic name for rerun entity path
            entity_path = f"camera{topic.replace('/', '_')}"
            
            # Log image to rerun
            rr.set_time_seconds("timestamp", timestamp)
            rr.log(entity_path, rr.Image(rgb_image))
            
        except Exception as e:
            print(f"Error processing image from topic {topic}: {str(e)}")
    
    def play_rosbag(self, bag_path: str, playback_speed: float = 1.0, sync_tolerance: float = 0.1, 
                    save_file: Optional[str] = None) -> None:
        """
        Play back all image topics from the rosbag with topic synchronization.
        
        Args:
            bag_path: Path to the rosbag file
            playback_speed: Playback speed multiplier (1.0 = real-time)
            sync_tolerance: Maximum time difference for synchronization in seconds
            save_file: Optional path to save rerun recording (.rrd file)
        """
        bag_path = Path(bag_path)
        if not bag_path.exists():
            raise FileNotFoundError(f"Rosbag file not found: {bag_path}")
            
        # Get all image topics
        self.image_topics = self._get_image_topics(str(bag_path))
        
        if not self.image_topics:
            print("No image topics found in the rosbag")
            return
            
        print(f"Found image topics: {self.image_topics}")
        print(f"Synchronization tolerance: {sync_tolerance}s")
        
        # Initialize rerun with optional recording
        if save_file:
            # Generate default filename if save_file is True or empty
            # if save_file == "True" or save_file == "":
            bag_name = bag_path.stem
            save_file = f"{bag_name}_rerun.rrd"
            
            print(f"Recording to: {save_file}")
            rr.init("rosbag_image_player", spawn=False)
            rr.save(save_file)
        else:
            rr.init("rosbag_image_player", spawn=True)
        
        # Use synchronized playback
        self._play_synchronized(bag_path, playback_speed, sync_tolerance)
        
        print("Rosbag playback completed")
    
    def _get_total_image_messages(self, bag_path: Path) -> int:
        """
        Count total image messages for progress tracking.
        
        Args:
            bag_path: Path to the rosbag file
            
        Returns:
            Total number of image messages
        """
        total_count = 0
        with AnyReader([bag_path]) as reader:
            for connection, _, _ in reader.messages():
                if connection.topic in self.image_topics:
                    total_count += 1
        return total_count
    
    def _play_synchronized(self, bag_path: Path, playback_speed: float, sync_tolerance: float) -> None:
        """
        Play back messages with streaming topic synchronization using message buffers.
        
        Args:
            bag_path: Path to the rosbag file
            playback_speed: Playback speed multiplier
            sync_tolerance: Maximum time difference for synchronization
        """
        # Get total messages for progress tracking
        print("Counting messages for progress tracking...")
        total_messages = self._get_total_image_messages(bag_path)
        
        # Message buffers for each topic (limited size)
        topic_buffers: Dict[str, deque] = {topic: deque(maxlen=100) for topic in self.image_topics}
        
        start_wall_time = time.time()
        start_bag_time = None
        last_sync_time = None
        processed_count = 0
        
        with AnyReader([bag_path]) as reader:
            print(f"Playing rosbag: {bag_path}")
            print(f"Playback speed: {playback_speed}x")
            print(f"Total image messages: {total_messages:,}")
            
            # Initialize progress bar
            with tqdm(total=total_messages, desc="Processing", unit="msgs", 
                     bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
                
                # Stream messages and process with synchronization
                for connection, timestamp, rawdata in reader.messages():
                    if connection.topic not in self.image_topics:
                        continue
                        
                    # Deserialize message
                    msg_data = reader.deserialize(rawdata, connection.msgtype)
                    current_time = timestamp / 1e9
                    
                    if start_bag_time is None:
                        start_bag_time = current_time
                    
                    # Add message to topic buffer
                    topic_buffers[connection.topic].append({
                        'timestamp': current_time,
                        'msg_data': msg_data,
                        'msg_type': connection.msgtype,
                        'topic': connection.topic
                    })
                    
                    # Update progress
                    processed_count += 1
                    pbar.update(1)
                    
                    # Update progress bar description with current time
                    pbar.set_description(f"Processing (t={current_time:.1f}s)")
                    
                    # Try to find synchronized messages
                    sync_groups = self._find_sync_groups(topic_buffers, sync_tolerance)
                    
                    # Process and remove synchronized groups
                    for sync_group in sync_groups:
                        if not sync_group:
                            continue
                            
                        # Calculate average sync time
                        sync_time = sum(msg['timestamp'] for msg in sync_group) / len(sync_group)
                        
                        # Wait for proper timing
                        if last_sync_time is not None:
                            elapsed_wall_time = time.time() - start_wall_time
                            expected_wall_time = (sync_time - start_bag_time) / playback_speed
                            
                            if elapsed_wall_time < expected_wall_time:
                                time.sleep(expected_wall_time - elapsed_wall_time)
                        
                        # Process synchronized group
                        # print(f"Sync group at {sync_time:.3f}s: {[msg['topic'] for msg in sync_group]}")
                        
                        for msg in sync_group:
                            self._process_image_message(
                                msg['msg_data'], 
                                msg['msg_type'], 
                                msg['topic'], 
                                sync_time  # Use synchronized time
                            )
                        
                        last_sync_time = sync_time
                        
                        # Remove processed messages from buffers
                        self._remove_processed_messages(topic_buffers, sync_group)
                
                # Process remaining messages in buffers
                pbar.set_description("Flushing remaining messages")
                self._flush_remaining_messages(topic_buffers, start_bag_time, start_wall_time, playback_speed, last_sync_time)
    
    def _find_sync_groups(self, topic_buffers: Dict[str, deque], sync_tolerance: float) -> List[List[Dict]]:
        """
        Find synchronized message groups from topic buffers.
        
        Args:
            topic_buffers: Message buffers for each topic
            sync_tolerance: Maximum time difference for synchronization
            
        Returns:
            List of synchronized message groups
        """
        sync_groups = []
        
        # Get all timestamps from buffers
        all_messages = []
        for topic, buffer in topic_buffers.items():
            for msg in buffer:
                all_messages.append(msg)
        
        if not all_messages:
            return sync_groups
            
        # Sort by timestamp
        all_messages.sort(key=lambda x: x['timestamp'])
        
        # Find groups that can be synchronized
        processed_topics = set()
        
        for i, msg in enumerate(all_messages):
            if msg['topic'] in processed_topics:
                continue
                
            sync_group = [msg]
            sync_time = msg['timestamp']
            group_topics = {msg['topic']}
            
            # Look for messages from other topics within tolerance
            for j in range(i + 1, len(all_messages)):
                candidate = all_messages[j]
                
                if candidate['topic'] in group_topics:
                    continue
                    
                time_diff = abs(candidate['timestamp'] - sync_time)
                
                if time_diff <= sync_tolerance:
                    sync_group.append(candidate)
                    group_topics.add(candidate['topic'])
                    # Update sync time to average
                    sync_time = sum(m['timestamp'] for m in sync_group) / len(sync_group)
                elif candidate['timestamp'] > sync_time + sync_tolerance:
                    # No more candidates for this group
                    break
            
            # Only process groups with multiple topics or if buffer is getting full
            if len(sync_group) > 1 or any(len(buf) > 50 for buf in topic_buffers.values()):
                sync_groups.append(sync_group)
                processed_topics.update(group_topics)
        
        return sync_groups
    
    def _remove_processed_messages(self, topic_buffers: Dict[str, deque], sync_group: List[Dict]) -> None:
        """
        Remove processed messages from topic buffers.
        
        Args:
            topic_buffers: Message buffers for each topic
            sync_group: Synchronized message group to remove
        """
        for msg in sync_group:
            topic = msg['topic']
            buffer = topic_buffers[topic]
            
            # Remove the specific message from buffer
            for i, buffered_msg in enumerate(buffer):
                if (buffered_msg['timestamp'] == msg['timestamp'] and 
                    buffered_msg['topic'] == msg['topic']):
                    del buffer[i]
                    break
    
    def _flush_remaining_messages(self, topic_buffers: Dict[str, deque], start_bag_time: float, 
                                 start_wall_time: float, playback_speed: float, last_sync_time: Optional[float]) -> None:
        """
        Process any remaining messages in buffers.
        
        Args:
            topic_buffers: Message buffers for each topic
            start_bag_time: Start time of bag playback
            start_wall_time: Wall clock start time
            playback_speed: Playback speed multiplier
            last_sync_time: Last synchronized timestamp
        """
        # Collect all remaining messages
        remaining_messages = []
        for topic, buffer in topic_buffers.items():
            remaining_messages.extend(list(buffer))
        
        # Sort and process remaining messages
        remaining_messages.sort(key=lambda x: x['timestamp'])
        
        for msg in remaining_messages:
            # Wait for proper timing
            if last_sync_time is not None:
                elapsed_wall_time = time.time() - start_wall_time
                expected_wall_time = (msg['timestamp'] - start_bag_time) / playback_speed
                
                if elapsed_wall_time < expected_wall_time:
                    time.sleep(expected_wall_time - elapsed_wall_time)
            
            # Process individual message
            self._process_image_message(
                msg['msg_data'], 
                msg['msg_type'], 
                msg['topic'], 
                msg['timestamp']
            )


def main():
    parser = argparse.ArgumentParser(description='Play image topics from rosbag (ROS1/ROS2) with synchronized rerun visualization')
    parser.add_argument('bag_path', help='Path to the rosbag file (ROS1) or directory (ROS2)')
    parser.add_argument('--speed', type=float, default=1.0, help='Playback speed multiplier (default: 1.0)')
    parser.add_argument('--max-resolution', type=int, default=1024, 
                       help='Maximum resolution for width or height (default: 1024)')
    parser.add_argument('--sync-tolerance', type=float, default=0.1,
                       help='Maximum time difference for topic synchronization in seconds (default: 0.1)')
    parser.add_argument('--save', action='store_true', 
                       help='Save rerun recording to file (.rrd). Use without value for auto-naming.')
    
    args = parser.parse_args()
    
    try:
        player = RosbagImagePlayer(max_resolution=args.max_resolution)
        player.play_rosbag(args.bag_path, args.speed, args.sync_tolerance, args.save)
    except KeyboardInterrupt:
        print("\nPlayback interrupted by user")
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()