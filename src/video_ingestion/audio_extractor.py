
import subprocess


def extract_audio(video_path: str, output_audio_path: str) -> None:
    """
    Extracts audio from an MP4 video file using ffmpeg.
    """
    command = [
        "ffmpeg",
        "-i", video_path,
        "-vn", # No video
        "-acodec", "pcm_s16le", # Audio codec
        "-ar", "16000", # Audio sample rate (suitable for speech)
        "-ac", "1", # Mono audio
        output_audio_path
    ]
    try:
        print(f"Attempting to extract audio from {video_path} to {output_audio_path} using ffmpeg...")
        subprocess.run(command, check=True, capture_output=True)
        print("Audio extraction successful.")
    except FileNotFoundError:
        raise RuntimeError("ffmpeg not found. Please install ffmpeg and ensure it's in your system's PATH.")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg audio extraction failed: {e.stderr.decode()}")
