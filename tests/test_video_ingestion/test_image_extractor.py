import pytest
import os
import subprocess
from unittest.mock import patch, MagicMock
from pathlib import Path
from src.video_ingestion.image_extractor import extract_screenshots

def test_extract_screenshots_success(tmp_path):
    """Test successful screenshot extraction."""
    # Create a dummy video file path
    video_path = tmp_path / "test_video.mp4"
    video_path.write_bytes(b"dummy video content")
    
    output_dir = tmp_path / "screenshots"
    
    # Mock successful ffmpeg execution
    with patch('subprocess.run') as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        
        # Create expected screenshot files to simulate ffmpeg output
        output_dir.mkdir(exist_ok=True)
        expected_files = [
            output_dir / "test_video_000s.jpg",
            output_dir / "test_video_004s.jpg", 
            output_dir / "test_video_008s.jpg"
        ]
        for file_path in expected_files:
            file_path.write_bytes(b"fake jpg content")
        
        result = extract_screenshots(str(video_path), str(output_dir), interval_seconds=4)
        
        # Verify ffmpeg was called with correct parameters
        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        assert "ffmpeg" in call_args
        assert "-i" in call_args
        assert str(video_path) in call_args
        assert "fps=1/4" in call_args
        
        # Verify returned files
        assert len(result) == 3
        assert all(os.path.exists(f) for f in result)
        assert "test_video_000s.jpg" in result[0]
        assert "test_video_004s.jpg" in result[1]
        assert "test_video_008s.jpg" in result[2]

def test_extract_screenshots_creates_output_directory(tmp_path):
    """Test that output directory is created if it doesn't exist."""
    video_path = tmp_path / "test_video.mp4" 
    video_path.write_bytes(b"dummy video content")
    
    output_dir = tmp_path / "nonexistent" / "screenshots"
    assert not output_dir.exists()
    
    with patch('subprocess.run') as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        
        extract_screenshots(str(video_path), str(output_dir), interval_seconds=4)
        
        # Verify directory was created
        assert output_dir.exists()
        assert output_dir.is_dir()

def test_extract_screenshots_ffmpeg_not_found():
    """Test error handling when ffmpeg is not found."""
    with patch('subprocess.run', side_effect=FileNotFoundError("ffmpeg not found")):
        with pytest.raises(RuntimeError, match="ffmpeg not found"):
            extract_screenshots("test.mp4", "output", interval_seconds=4)

def test_extract_screenshots_ffmpeg_failure(tmp_path):
    """Test error handling when ffmpeg command fails."""
    video_path = tmp_path / "test_video.mp4"
    video_path.write_bytes(b"dummy content")
    
    mock_error = subprocess.CalledProcessError(1, "ffmpeg")
    mock_error.stderr = "Invalid input file"
    
    with patch('subprocess.run', side_effect=mock_error):
        with pytest.raises(RuntimeError, match="ffmpeg screenshot extraction failed"):
            extract_screenshots(str(video_path), str(tmp_path), interval_seconds=4)

def test_extract_screenshots_custom_interval(tmp_path):
    """Test screenshot extraction with custom interval."""
    video_path = tmp_path / "test_video.mp4"
    video_path.write_bytes(b"dummy content")
    output_dir = tmp_path / "screenshots"
    
    with patch('subprocess.run') as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        
        # Create expected files for 2-second intervals
        output_dir.mkdir(exist_ok=True)
        expected_files = [
            output_dir / "test_video_000s.jpg",
            output_dir / "test_video_002s.jpg",
            output_dir / "test_video_004s.jpg",
            output_dir / "test_video_006s.jpg"
        ]
        for file_path in expected_files:
            file_path.write_bytes(b"fake jpg content")
        
        result = extract_screenshots(str(video_path), str(output_dir), interval_seconds=2)
        
        # Verify correct fps parameter for 2-second intervals
        call_args = mock_run.call_args[0][0]
        assert "fps=1/2" in call_args
        
        # Verify returned files
        assert len(result) == 4

def test_extract_screenshots_filename_format(tmp_path):
    """Test that screenshot filenames follow the correct pattern."""
    video_path = tmp_path / "my_trading_video.mp4"
    video_path.write_bytes(b"dummy content")
    output_dir = tmp_path / "screenshots"
    
    with patch('subprocess.run') as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        
        # Create a screenshot file with the expected naming pattern
        output_dir.mkdir(exist_ok=True)
        expected_file = output_dir / "my_trading_video_000s.jpg"
        expected_file.write_bytes(b"fake jpg content")
        
        result = extract_screenshots(str(video_path), str(output_dir), interval_seconds=4)
        
        # Verify ffmpeg output pattern uses correct video name
        call_args = mock_run.call_args[0][0]
        output_pattern = call_args[-1]  # Last argument should be output pattern
        assert "my_trading_video_%03ds.jpg" in output_pattern

def test_extract_screenshots_empty_result(tmp_path):
    """Test when no screenshots are generated."""
    video_path = tmp_path / "test_video.mp4"
    video_path.write_bytes(b"dummy content")
    output_dir = tmp_path / "screenshots"
    
    with patch('subprocess.run') as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        
        # Don't create any screenshot files to simulate no output
        output_dir.mkdir(exist_ok=True)
        
        result = extract_screenshots(str(video_path), str(output_dir), interval_seconds=4)
        
        # Should return empty list
        assert result == []

def test_extract_screenshots_video_stem_extraction(tmp_path):
    """Test that video filename stem is correctly extracted."""
    # Test with various file extensions and paths
    test_cases = [
        ("video.mp4", "video"),
        ("my_video.mov", "my_video"), 
        ("path/to/trading_analysis.avi", "trading_analysis"),
        ("complex.name.with.dots.mp4", "complex.name.with.dots")
    ]
    
    for video_filename, expected_stem in test_cases:
        video_path = tmp_path / video_filename
        video_path.parent.mkdir(parents=True, exist_ok=True)
        video_path.write_bytes(b"dummy content")
        output_dir = tmp_path / "screenshots"
        
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            
            extract_screenshots(str(video_path), str(output_dir), interval_seconds=4)
            
            # Verify the output pattern uses the correct stem
            call_args = mock_run.call_args[0][0]
            output_pattern = call_args[-1]
            assert f"{expected_stem}_%03ds.jpg" in output_pattern