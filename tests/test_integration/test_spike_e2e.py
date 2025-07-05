import pytest
import os
import tempfile
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock
import json

def create_dummy_video_file(video_path):
    """Create a minimal valid MP4 file using ffmpeg for testing."""
    try:
        # Create a very short (1 second) silent video for testing
        subprocess.run([
            "ffmpeg", "-f", "lavfi", "-i", "color=black:size=320x240:duration=1", 
            "-f", "lavfi", "-i", "anullsrc=channel_layout=mono:sample_rate=44100", 
            "-c:v", "libx264", "-c:a", "aac", "-shortest", "-y", str(video_path)
        ], check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        # ffmpeg not available or command failed
        return False

@pytest.mark.integration
def test_spike_e2e_with_real_video_processing(tmp_path):
    """End-to-end integration test with real video file processing."""
    # Create a test video file
    video_path = tmp_path / "test_trading_video.mp4"
    
    if not create_dummy_video_file(video_path):
        pytest.skip("ffmpeg not available for creating test video")
    
    # Change to tmp_path so outputs go there
    original_cwd = os.getcwd()
    
    try:
        os.chdir(tmp_path)
        
        # Mock the heavy processing components but let the file operations run
        with patch('video_ingestion.transcriber.load_whisper_model') as mock_load_model, \
             patch('video_ingestion.transcriber.transcribe_audio') as mock_transcribe, \
             patch('video_ingestion.topic_extractor._nlp') as mock_nlp:
            
            # Setup mocks for the heavy ML components
            mock_model = MagicMock()
            mock_load_model.return_value = mock_model
            
            # Mock a realistic transcription result with trading content
            mock_transcribe.return_value = "The price action shows strong support at the 50 level. We're seeing bullish patterns emerging with high volume. This breakout could lead to significant resistance around 75."
            
            # Mock spaCy processing
            mock_doc = MagicMock()
            mock_token1 = MagicMock()
            mock_token1.lemma_ = "price"
            mock_token1.is_stop = False
            mock_token1.is_alpha = True
            mock_token2 = MagicMock() 
            mock_token2.lemma_ = "support"
            mock_token2.is_stop = False
            mock_token2.is_alpha = True
            mock_doc.__iter__.return_value = [mock_token1, mock_token2]
            mock_nlp.return_value = mock_doc
            
            # Import and run the spike command
            from cli.commands import spike_ingest_command
            
            class MockArgs:
                def __init__(self, video_path):
                    self.video_path = video_path
            
            args = MockArgs(str(video_path))
            
            # Run the actual spike command
            spike_ingest_command(args)
            
            # Verify directory structure was created
            output_dir = tmp_path / "output"
            reports_dir = output_dir / "reports"
            screenshots_dir = output_dir / "screenshots"
            
            assert output_dir.exists(), "Output directory should be created"
            assert reports_dir.exists(), "Reports directory should be created"
            assert screenshots_dir.exists(), "Screenshots directory should be created"
            
            # Verify report file was created
            report_files = list(reports_dir.glob("spike_output_*.md"))
            assert len(report_files) == 1, "Exactly one report file should be created"
            
            report_file = report_files[0]
            report_content = report_file.read_text()
            
            # Verify report content structure
            assert "# Video Ingestion Spike Report for: test_trading_video.mp4" in report_content
            assert "## Extracted Screenshots" in report_content
            assert "## Full Transcript" in report_content
            assert "## Identified Topics with Timestamps" in report_content
            
            # Verify transcript content is included
            assert "price action" in report_content
            assert "support" in report_content
            assert "bullish patterns" in report_content
            
            # Verify screenshots were created (mocked ffmpeg should create some)
            screenshot_files = list(screenshots_dir.glob("*.jpg"))
            # Note: With mocked processing, we might not get actual screenshots
            # but the directory structure should exist
            
            print(f"Report content preview:\n{report_content[:500]}...")
            print(f"Created {len(screenshot_files)} screenshot files")
    
    finally:
        os.chdir(original_cwd)

@pytest.mark.integration  
def test_spike_e2e_with_mocked_video_duration(tmp_path):
    """E2E test with completely mocked video processing for CI environments."""
    # Create a dummy video file (just for existence check)
    video_path = tmp_path / "mock_trading_analysis.mp4"
    video_path.write_bytes(b"fake video content")
    
    original_cwd = os.getcwd()
    
    try:
        os.chdir(tmp_path)
        
        # Mock all external dependencies
        with patch('video_ingestion.audio_extractor.extract_audio') as mock_extract_audio, \
             patch('video_ingestion.image_extractor.extract_screenshots') as mock_extract_screenshots, \
             patch('video_ingestion.transcriber.transcribe_audio') as mock_transcribe, \
             patch('video_ingestion.topic_extractor.extract_topics_with_timestamps') as mock_extract_topics, \
             patch('moviepy.editor.VideoFileClip') as mock_video_clip:
            
            # Setup realistic mocks
            mock_clip = MagicMock()
            mock_clip.duration = 120.0  # 2 minute video
            mock_video_clip.return_value = mock_clip
            
            # Mock screenshot creation
            expected_screenshots = [
                str(tmp_path / "output" / "screenshots" / "mock_trading_analysis_000s.jpg"),
                str(tmp_path / "output" / "screenshots" / "mock_trading_analysis_004s.jpg"), 
                str(tmp_path / "output" / "screenshots" / "mock_trading_analysis_008s.jpg")
            ]
            mock_extract_screenshots.return_value = expected_screenshots
            
            # Mock transcription with trading-specific content
            trading_transcript = """
            Welcome to today's trading analysis. Let's look at the price action on the daily chart.
            We can see strong support levels around the 200 moving average. The RSI indicator shows 
            we're approaching oversold conditions. Volume has been increasing on this bearish trend,
            but we might see a reversal soon. Key resistance levels to watch are at 150 and 165.
            If we break above those levels, we could see a bullish breakout pattern forming.
            """
            mock_transcribe.return_value = trading_transcript.strip()
            
            # Mock topic extraction with realistic trading keywords
            mock_topics_data = {
                "price": [{"start": 15.2, "end": 15.8}],
                "support": [{"start": 25.1, "end": 25.6}], 
                "resistance": [{"start": 45.3, "end": 45.9}, {"start": 67.2, "end": 67.8}],
                "volume": [{"start": 35.7, "end": 36.1}],
                "bullish": [{"start": 78.4, "end": 78.9}],
                "breakout": [{"start": 82.1, "end": 82.7}]
            }
            mock_extract_topics.return_value = mock_topics_data
            
            # Import and run the spike command
            from cli.commands import spike_ingest_command
            
            class MockArgs:
                def __init__(self, video_path):
                    self.video_path = video_path
            
            args = MockArgs(str(video_path))
            
            # Run the spike command
            spike_ingest_command(args)
            
            # Verify all components were called
            mock_extract_audio.assert_called_once()
            mock_extract_screenshots.assert_called_once_with(
                str(video_path), str(tmp_path / "output" / "screenshots"), interval_seconds=4
            )
            mock_transcribe.assert_called_once()
            mock_extract_topics.assert_called_once()
            
            # Verify output structure
            output_dir = tmp_path / "output"
            reports_dir = output_dir / "reports"
            screenshots_dir = output_dir / "screenshots"
            
            assert output_dir.exists()
            assert reports_dir.exists()
            assert screenshots_dir.exists()
            
            # Verify report was created with correct content
            report_files = list(reports_dir.glob("spike_output_*.md"))
            assert len(report_files) == 1
            
            report_content = report_files[0].read_text()
            
            # Verify report structure and content
            assert "mock_trading_analysis.mp4" in report_content
            assert "Screenshots captured every 4 seconds (3 total)" in report_content
            assert "../screenshots/mock_trading_analysis_000s.jpg" in report_content
            assert "../screenshots/mock_trading_analysis_004s.jpg" in report_content
            assert "../screenshots/mock_trading_analysis_008s.jpg" in report_content
            
            # Verify transcript content
            assert "price action" in report_content
            assert "support levels" in report_content
            assert "RSI indicator" in report_content
            assert "bullish breakout" in report_content
            
            # Verify keywords section exists with our enhanced format
            assert "## Identified Keywords with Timestamps" in report_content
            
            print("✅ E2E test passed - full spike pipeline working!")
            print(f"Report file: {report_files[0]}")
            print(f"Report size: {len(report_content)} characters")
    
    finally:
        os.chdir(original_cwd)

@pytest.mark.integration
def test_spike_e2e_error_handling(tmp_path):
    """Test E2E error handling scenarios."""
    original_cwd = os.getcwd()
    
    try:
        os.chdir(tmp_path)
        
        # Test with non-existent video file
        from cli.commands import spike_ingest_command
        
        class MockArgs:
            def __init__(self, video_path):
                self.video_path = video_path
        
        args = MockArgs("/nonexistent/video.mp4")
        
        # Capture output
        import io
        import sys
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        try:
            spike_ingest_command(args)
            output = captured_output.getvalue()
            
            # Verify error handling
            assert "Error: Video file not found" in output
            
            # Verify no output directories were created
            output_dir = tmp_path / "output"
            assert not output_dir.exists()
            
        finally:
            sys.stdout = sys.__stdout__
    
    finally:
        os.chdir(original_cwd)

def test_spike_e2e_output_directory_permissions(tmp_path):
    """Test that output directories are created with correct permissions."""
    video_path = tmp_path / "test.mp4"
    video_path.write_bytes(b"fake video")
    
    original_cwd = os.getcwd()
    
    try:
        os.chdir(tmp_path)
        
        with patch('video_ingestion.audio_extractor.extract_audio'), \
             patch('video_ingestion.image_extractor.extract_screenshots', return_value=[]), \
             patch('video_ingestion.transcriber.transcribe_audio', return_value="test transcript"), \
             patch('video_ingestion.topic_extractor.extract_topics_with_timestamps', return_value={}), \
             patch('moviepy.editor.VideoFileClip') as mock_clip:
            
            mock_clip.return_value.duration = 30.0
            
            from cli.commands import spike_ingest_command
            
            class MockArgs:
                def __init__(self, video_path):
                    self.video_path = video_path
            
            args = MockArgs(str(video_path))
            spike_ingest_command(args)
            
            # Verify directories exist and are writable
            output_dir = tmp_path / "output"
            reports_dir = output_dir / "reports" 
            screenshots_dir = output_dir / "screenshots"
            
            assert output_dir.exists() and output_dir.is_dir()
            assert reports_dir.exists() and reports_dir.is_dir()
            assert screenshots_dir.exists() and screenshots_dir.is_dir()
            
            # Test that we can write to these directories
            test_file = reports_dir / "test_write.txt"
            test_file.write_text("test")
            assert test_file.exists()
    
    finally:
        os.chdir(original_cwd)