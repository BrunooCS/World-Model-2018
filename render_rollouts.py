import h5py
import numpy as np
import cv2
from tqdm import tqdm
import os
import argparse

def save_episode_as_video(h5_path, episode_idx, output_dir='videos', fps=30, resolution=None):
    """
    Extract an episode from the HDF5 dataset and save it as an MP4 video.
    
    Args:
        h5_path (str): Path to the HDF5 dataset
        episode_idx (int): Index of the episode to extract
        output_dir (str): Directory to save the video
        fps (int): Frames per second for the output video
        resolution (tuple): Optional resolution to resize frames (width, height)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Open the HDF5 file and extract the episode
    with h5py.File(h5_path, 'r') as h5f:
        # Check if the episode index is valid
        num_episodes = h5f['images'].shape[0]
        if episode_idx >= num_episodes:
            raise ValueError(f"Episode index {episode_idx} out of range (0-{num_episodes-1})")
        
        # Extract all frames for this episode
        print(f"Extracting episode {episode_idx} from dataset...")
        images = h5f['images'][episode_idx]
        actions = h5f['actions'][episode_idx]
        rewards = h5f['rewards'][episode_idx]
    
    # Define the output video path
    video_path = os.path.join(output_dir, f"episode_{episode_idx}.mp4")
    
    # Get frame dimensions
    height, width, channels = images[0].shape
    if resolution:
        width, height = resolution
    
    # Initialize the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    
    # Write each frame to the video
    print(f"Creating video for episode {episode_idx}...")
    for i, frame in enumerate(tqdm(images)):
        # Add action and reward info as text overlay
        frame_with_info = frame.copy()
        
        # Convert to BGR for OpenCV
        frame_with_info = cv2.cvtColor(frame_with_info, cv2.COLOR_RGB2BGR)
        
        # Resize if needed
        if resolution:
            frame_with_info = cv2.resize(frame_with_info, (width, height), interpolation=cv2.INTER_AREA)
        
        # Write to video
        video.write(frame_with_info)
    
    # Release the video writer
    video.release()
    print(f"Video saved to {video_path}")

def generate_multiple_videos(h5_path, num_videos=5, output_dir='videos', fps=30):
    """
    Generate videos for multiple episodes.
    
    Args:
        h5_path (str): Path to the HDF5 dataset
        num_videos (int): Number of videos to generate
        output_dir (str): Directory to save videos
        fps (int): Frames per second for the output videos
    """
    # Open the HDF5 file to get the total number of episodes
    with h5py.File(h5_path, 'r') as h5f:
        total_episodes = h5f['images'].shape[0]
    
    # Choose episode indices at even intervals
    episode_indices = np.linspace(0, total_episodes-1, num_videos, dtype=int)
    
    # Generate a video for each selected episode
    for idx in episode_indices:
        save_episode_as_video(h5_path, idx, output_dir, fps)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate MP4 videos from HDF5 dataset episodes")
    parser.add_argument("--h5_path", type=str, default="car_racing_data_128.h5", help="Path to the HDF5 dataset")
    parser.add_argument("--num_videos", type=int, default=16, help="Number of videos to generate")
    parser.add_argument("--output_dir", type=str, default="videos", help="Directory to save videos")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second for the output videos")
    
    args = parser.parse_args()
    generate_multiple_videos(args.h5_path, args.num_videos, args.output_dir, args.fps)