import pytest
import os
import tempfile
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path
import uuid
import datetime
from cli.commands import spike_ingest_command

class MockArgs:
    """Mock args object for testing CLI commands."""
    def __init__(self, video_path):
        self.video_path = video_path

def test_spike_ingest_command_creates_output_directory_structure(tmp_path):
    """Test that spike command creates proper output directory structure."""
    video_path = tmp_path / "test_video.mp4"
    video_path.write_bytes(b"dummy video content")
    
    args = MockArgs(str(video_path))
    
    # Mock all the dependencies
    with patch('cli.commands.VideoFileClip') as mock_clip_class, \
         patch('cli.commands.extract_audio') as mock_extract_audio, \
         patch('cli.commands.transcribe_audio') as mock_transcribe, \
         patch('cli.commands.extract_topics_with_timestamps') as mock_extract_topics, \
         patch('cli.commands.os.getcwd', return_value=str(tmp_path)), \
         patch('builtins.open', mock_open()) as mock_file:
        
        # Setup mocks
        mock_clip = MagicMock()
        mock_clip.duration = 60.0
        mock_clip_class.return_value = mock_clip
        mock_transcribe.return_value = "This is a test transcript"
        mock_extract_topics.return_value = {}
        
        # Run the command
        spike_ingest_command(args)
        
        # Verify output directories were created
        output_dir = tmp_path / "output"
        reports_dir = tmp_path / "output" / "reports"
        screenshots_dir = tmp_path / "output" / "screenshots"
        
        assert output_dir.exists()
        assert reports_dir.exists() 
        assert screenshots_dir.exists()

def test_spike_ingest_command_saves_report_to_reports_directory(tmp_path):
    """Test that the markdown report is saved to output/reports/ directory."""
    video_path = tmp_path / "test_video.mp4" 
    video_path.write_bytes(b"dummy video content")
    
    args = MockArgs(str(video_path))
    
    with patch('cli.commands.VideoFileClip') as mock_clip_class, \
         patch('cli.commands.extract_audio') as mock_extract_audio, \
         patch('cli.commands.extract_screenshots') as mock_extract_screenshots, \
         patch('cli.commands.transcribe_audio') as mock_transcribe, \
         patch('cli.commands.extract_topics_with_timestamps') as mock_extract_topics, \
         patch('cli.commands.os.getcwd', return_value=str(tmp_path)), \
         patch('builtins.open', mock_open()) as mock_file, \
         patch('cli.commands.uuid.uuid4') as mock_uuid:
        
        # Setup mocks
        mock_clip = MagicMock()
        mock_clip.duration = 60.0
        mock_clip_class.return_value = mock_clip
        mock_extract_screenshots.return_value = []
        mock_transcribe.return_value = "This is a test transcript"
        mock_extract_topics.return_value = {}
        mock_uuid.return_value.hex = "12345678abcdef"
        
        # Run the command
        spike_ingest_command(args)
        
        # Verify the file was opened with the correct path in reports directory
        expected_path = str(tmp_path / "output" / "reports" / "spike_output_12345678.md")
        mock_file.assert_called_with(expected_path, "w")

def test_spike_ingest_command_integrates_image_extraction(tmp_path):
    """Test that the spike command calls image extraction and includes screenshots in report."""
    video_path = tmp_path / "test_video.mp4"
    video_path.write_bytes(b"dummy video content")
    
    args = MockArgs(str(video_path))
    
    with patch('cli.commands.VideoFileClip') as mock_clip_class, \
         patch('cli.commands.extract_audio') as mock_extract_audio, \
         patch('cli.commands.extract_screenshots') as mock_extract_screenshots, \
         patch('cli.commands.transcribe_audio') as mock_transcribe, \
         patch('cli.commands.extract_topics_with_timestamps') as mock_extract_topics, \
         patch('cli.commands.os.getcwd', return_value=str(tmp_path)), \
         patch('builtins.open', mock_open()) as mock_file:
        
        # Setup mocks
        mock_clip = MagicMock()
        mock_clip.duration = 60.0
        mock_clip_class.return_value = mock_clip
        mock_transcribe.return_value = "This is a test transcript"
        mock_extract_topics.return_value = {}
        mock_extract_screenshots.return_value = [
            str(tmp_path / "output" / "screenshots" / "test_video_000s.jpg"),
            str(tmp_path / "output" / "screenshots" / "test_video_004s.jpg")
        ]
        
        # Run the command
        spike_ingest_command(args)
        
        # Verify extract_screenshots was called with correct parameters
        expected_screenshots_dir = str(tmp_path / "output" / "screenshots")
        mock_extract_screenshots.assert_called_once_with(
            str(video_path), expected_screenshots_dir, interval_seconds=4
        )

def test_spike_ingest_command_report_includes_screenshots_section(tmp_path):
    """Test that the generated report includes a screenshots section."""
    video_path = tmp_path / "test_video.mp4"
    video_path.write_bytes(b"dummy video content")
    
    args = MockArgs(str(video_path))
    
    # Capture what gets written to the file
    file_content = []
    def mock_write(content):
        file_content.append(content)
    
    with patch('cli.commands.VideoFileClip') as mock_clip_class, \
         patch('cli.commands.extract_audio') as mock_extract_audio, \
         patch('cli.commands.extract_screenshots') as mock_extract_screenshots, \
         patch('cli.commands.transcribe_audio') as mock_transcribe, \
         patch('cli.commands.extract_topics_with_timestamps') as mock_extract_topics, \
         patch('cli.commands.os.getcwd', return_value=str(tmp_path)), \
         patch('builtins.open', mock_open()) as mock_file:
        
        # Setup mocks
        mock_clip = MagicMock()
        mock_clip.duration = 60.0
        mock_clip_class.return_value = mock_clip
        mock_transcribe.return_value = "This is a test transcript"
        mock_extract_topics.return_value = {}
        mock_extract_screenshots.return_value = [
            str(tmp_path / "output" / "screenshots" / "test_video_000s.jpg"),
            str(tmp_path / "output" / "screenshots" / "test_video_004s.jpg")
        ]
        
        # Mock file write to capture content
        mock_file.return_value.write.side_effect = mock_write
        
        # Run the command
        spike_ingest_command(args)
        
        # Combine all written content
        full_content = "".join(file_content)
        
        # Verify screenshots section is included
        assert "## Extracted Screenshots" in full_content
        assert "Screenshots captured every 4 seconds (2 total):" in full_content
        assert "../screenshots/test_video_000s.jpg" in full_content
        assert "../screenshots/test_video_004s.jpg" in full_content

def test_spike_ingest_command_prints_updated_success_messages(tmp_path):
    """Test that success messages include all output paths."""
    video_path = tmp_path / "test_video.mp4"
    video_path.write_bytes(b"dummy video content")
    
    args = MockArgs(str(video_path))
    
    with patch('cli.commands.VideoFileClip') as mock_clip_class, \
         patch('cli.commands.extract_audio') as mock_extract_audio, \
         patch('cli.commands.extract_screenshots') as mock_extract_screenshots, \
         patch('cli.commands.transcribe_audio') as mock_transcribe, \
         patch('cli.commands.extract_topics_with_timestamps') as mock_extract_topics, \
         patch('cli.commands.os.getcwd', return_value=str(tmp_path)), \
         patch('builtins.open', mock_open()) as mock_file, \
         patch('builtins.print') as mock_print:
        
        # Setup mocks
        mock_clip = MagicMock()
        mock_clip.duration = 60.0
        mock_clip_class.return_value = mock_clip
        mock_transcribe.return_value = "This is a test transcript"
        mock_extract_topics.return_value = {}
        mock_extract_screenshots.return_value = []
        
        # Run the command
        spike_ingest_command(args)
        
        # Verify success messages include all paths
        print_calls = [str(call) for call in mock_print.call_args_list]
        print_output = " ".join(print_calls)
        
        assert "Spike successful!" in print_output
        assert "Report generated at:" in print_output
        assert "Screenshots saved to:" in print_output
        assert "All outputs saved to:" in print_output
        assert "output/reports/" in print_output
        assert "output/screenshots" in print_output

def test_spike_ingest_command_handles_missing_video_file():
    """Test error handling when video file doesn't exist."""
    args = MockArgs("/nonexistent/video.mp4")
    
    with patch('builtins.print') as mock_print:
        spike_ingest_command(args)
        
        # Verify error message
        mock_print.assert_called_with("Error: Video file not found at /nonexistent/video.mp4")

def test_spike_ingest_command_handles_video_duration_error(tmp_path):
    """Test error handling when video duration cannot be determined."""
    video_path = tmp_path / "test_video.mp4"
    video_path.write_bytes(b"dummy video content")
    
    args = MockArgs(str(video_path))
    
    with patch('cli.commands.VideoFileClip', side_effect=Exception("Invalid video file")), \
         patch('builtins.print') as mock_print:
        
        spike_ingest_command(args)
        
        # Verify error message
        print_calls = [str(call) for call in mock_print.call_args_list]
        print_output = " ".join(print_calls)
        assert "Error getting video duration" in print_output
        assert "Cannot proceed with timestamping" in print_output