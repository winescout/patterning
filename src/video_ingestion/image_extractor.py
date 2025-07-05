import subprocess
import os
from pathlib import Path
from typing import List

def extract_screenshots(video_path: str, output_dir: str, interval_seconds: int = 4) -> List[str]:
    """
    Extracts screenshots from an MP4 video file every N seconds using ffmpeg.
    Returns list of generated screenshot filenames.
    
    Args:
        video_path: Path to the input video file
        output_dir: Directory to save screenshots
        interval_seconds: Time interval between screenshots (default: 4 seconds)
        
    Returns:
        List of paths to generated screenshot files
        
    Raises:
        RuntimeError: If ffmpeg is not found or extraction fails
    """
    video_name = Path(video_path).stem  # Get filename without extension
    os.makedirs(output_dir, exist_ok=True)
    
    # ffmpeg command to extract frames every N seconds
    # Format: video_name_000s.jpg, video_name_004s.jpg, etc.
    output_pattern = os.path.join(output_dir, f"{video_name}_%03ds.jpg")
    
    command = [
        "ffmpeg",
        "-i", video_path,
        "-vf", f"fps=1/{interval_seconds}",  # Extract 1 frame every N seconds
        "-y",  # Overwrite output files
        output_pattern
    ]
    
    try:
        print(f"Extracting screenshots from {video_path} every {interval_seconds} seconds...")
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        
        # Find all generated screenshot files
        screenshot_files = []
        for i in range(0, 3600, interval_seconds):  # Support up to 1 hour of video
            screenshot_path = os.path.join(output_dir, f"{video_name}_{i:03d}s.jpg")
            if os.path.exists(screenshot_path):
                screenshot_files.append(screenshot_path)
        
        print(f"Successfully extracted {len(screenshot_files)} screenshots")
        return screenshot_files
        
    except FileNotFoundError:
        raise RuntimeError("ffmpeg not found. Please install ffmpeg and ensure it's in your system's PATH.")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg screenshot extraction failed: {e.stderr}")