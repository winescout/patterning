
import pytest
import os
import subprocess
from src.video_ingestion.audio_extractor import extract_audio

def test_extract_audio_success(tmp_path):
    """Test successful audio extraction."""
    dummy_video_path = tmp_path / "dummy_video.mp4"
    # Create a very small, silent MP4 file using ffmpeg for testing
    # Create a very small, silent WAV file using ffmpeg for testing
    subprocess.run([
        "ffmpeg", "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=16000:duration=1",
        "-c:a", "pcm_s16le", str(dummy_video_path)
    ], check=True, capture_output=True)

    output_audio_path = tmp_path / "output_audio.wav"
    extract_audio(str(dummy_video_path), str(output_audio_path))
    assert os.path.exists(output_audio_path)
    assert os.path.getsize(output_audio_path) > 0

def test_extract_audio_invalid_path():
    """Test audio extraction with an invalid video path."""
    with pytest.raises(RuntimeError, match="ffmpeg audio extraction failed"):
        extract_audio("/non/existent/video.mp4", "output.wav")

def test_extract_audio_ffmpeg_not_found(monkeypatch):
    """Test audio extraction when ffmpeg is not found."""
    def mock_run(*args, **kwargs):
        raise FileNotFoundError("ffmpeg not found")
    monkeypatch.setattr(subprocess, "run", mock_run)

    with pytest.raises(RuntimeError, match="ffmpeg not found"):
        extract_audio("/path/to/video.mp4", "output.wav")
