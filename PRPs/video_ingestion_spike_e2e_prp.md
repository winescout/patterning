name: "Video Ingestion Spike: End-to-End Validation"description: |

## Purpose
This PRP outlines a focused spike to perform an end-to-end validation of the core technologies required for video ingestion: `ffmpeg` for audio extraction, the Video-Llama model for actual transcription, and NLP techniques for topic extraction. The output will be a markdown file containing the transcript and identified topics with timestamps. This spike aims to confirm the full pipeline's functionality and identify any practical limitations or setup challenges.

## Core Principles
1. **Context is King**: Include ALL necessary documentation, examples, and caveats
2. **Validation Loops**: Provide executable tests/lints the AI can run and fix
3. **Information Dense**: Use keywords and patterns from the codebase
4. **Progressive Success**: Start simple, validate, then enhance
5. **Global rules**: Be sure to follow all rules in GEMINI.md

---

## Goal
To successfully extract audio from an MP4 video file, transcribe it using a *real* Whisper model, extract topics from the transcription, capture screenshots every 4 seconds with timestamped filenames, and output the full transcript along with a list of topics and their corresponding timestamps to a markdown file. This will validate the entire core ingestion pipeline and confirm the user's system is capable of running these processes.

## Why
- To confirm that `ffmpeg` is correctly installed and efficiently extracts audio and captures video frames.
- To verify that the Whisper model can be successfully loaded, configured, and used for actual transcription on the user's system, assessing its performance and resource consumption.
- To validate the basic functionality of topic extraction from a real transcript.
- To test video frame extraction capabilities for visual analysis and thumbnails.
- To ensure the user has all necessary software and configurations in place for full feature development.
- To understand the quality and format of the transcription and topic extraction output from Whisper.

## What
This spike will implement a functional subset of the video ingestion pipeline:
- Accepting a local file path to an MP4 video as input.
- Extracting the audio track from the video using `ffmpeg`.
- Capturing video screenshots every 4 seconds using `ffmpeg`, saving them with descriptive filenames that include the original video name and timestamp (e.g., `video_name_004s.jpg`, `video_name_008s.jpg`).
- Loading and using the Whisper model to transcribe the extracted audio.
- Processing the transcription to identify specific keywords (e.g., price action, technical analysis terms) and their exact timestamps.
- Generating a markdown file containing the full transcript, a structured list of identified topics with their start and end timestamps, and references to captured screenshots.
- Providing clear feedback on the success or failure of each step, especially regarding external tool/model setup and execution.

### Success Criteria
- [ ] User can run a CLI command to initiate the spike.
- [ ] `ffmpeg` successfully extracts audio from a provided MP4 video.
- [ ] `ffmpeg` successfully captures screenshots every 4 seconds with timestamped filenames.
- [ ] Screenshot filenames follow the pattern: `{video_name}_{timestamp}.jpg` (e.g., `test_video_004s.jpg`).
- [ ] The Whisper model is successfully loaded and performs actual transcription.
- [ ] Specific keywords and their timestamps are successfully extracted from the transcription.
- [ ] A markdown file is generated with the video's transcript, structured list of topics/timestamps, and screenshot references.
- [ ] Clear error messages are provided if `ffmpeg` or Whisper operations fail.
- [ ] The generated markdown file is readable and contains the expected information.

## All Needed Context

### Documentation & References (list all context needed to implement the feature)
```yaml
# MUST READ - Include these in your context window
- url: https://github.com/openai/whisper
  why: Official GitHub repository for OpenAI Whisper. CRITICAL for installation, model download, and understanding its Python API for transcription.

- url: https://github.com/openai/whisper#available-models-and-languages
  why: Whisper documentation on available models and their sizes.

- url: https://ffmpeg.org/download.html
  why: Official download page for ffmpeg. Crucial for user installation instructions.

- url: https://ffmpeg.org/documentation.html
  why: Official documentation for ffmpeg, specifically for audio extraction from video.

- url: https://pydub.com/
  why: Python library for audio manipulation, useful for handling extracted audio.

- url: https://spacy.io/usage/linguistic-features#sbd
  why: spaCy documentation for Sentence Boundary Detection (SBD), useful for segmenting transcripts.

- url: https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction
  why: Documentation on text feature extraction (e.g., TF-IDF) for keyword extraction in topic identification.

- file: GEMINI.md
  why: Project-specific guidelines and AI behavior rules.

- file: INITIAL.md
  why: Overall project overview and high-level requirements.

- file: PRPs/video_ingestion_prp.md
  why: The full PRP that this spike is based on, providing broader context.
```

### Current Codebase tree (run `tree` in the root of the project) to get an overview of the codebase
```bash
.
├── .gitattributes
├── .gitignore
├── GEMINI.md
├── INITIAL.md
├── LICENSE
├── README.md
├── .gemini/
│   ├── settings.local.json
│   └── commands/
│       ├── execute-prp.md
│       └── generate-prp.md
├── .git/...
├── examples/
│   └── .gitkeep
└── PRPs/
    ├── video_ingestion_prp.md
    ├── video_clip_clustering_prp.md
    └── templates/
        └── prp_base.md
```

### Desired Codebase tree with files to be added and responsibility of file
```bash
.
├── src/
│   ├── __init__.py
│   ├── main.py             # CLI entry point, orchestrates spike
│   ├── video_ingestion/
│   │   ├── __init__.py
│   │   ├── audio_extractor.py # Handles audio extraction using ffmpeg/pydub
│   │   ├── image_extractor.py # Handles screenshot extraction using ffmpeg
│   │   ├── transcriber.py  # Handles transcription using Whisper model
│   │   └── topic_extractor.py # Handles topic identification and timestamping
│   └── cli/
│       ├── __init__.py
│       └── commands.py     # Defines CLI command for spike
├── tests/
│   ├── __init__.py
│   ├── test_video_ingestion/
│   │   ├── __init__.py
│   │   ├── test_audio_extractor.py
│   │   ├── test_image_extractor.py
│   │   ├── test_transcriber.py
│   │   └── test_topic_extractor.py
├── output/
│   ├── screenshots/        # Directory for extracted screenshots
│   └── reports/           # Directory for generated markdown reports
├── .env.example            # Example for model paths and other environment variables
```

### Known Gotchas of our codebase & Library Quirks
```python
# CRITICAL: ffmpeg MUST be installed and accessible in the system's PATH. This is the primary point of failure for audio extraction.
# CRITICAL: Whisper models are typically very large and require significant computational resources (e.g., GPU, large RAM). Users MUST ensure they have the necessary hardware and have downloaded the model weights.
# CRITICAL: The exact method for loading and using Whisper will depend on the specific version and its Python API. The user will need to follow the Whisper GitHub instructions carefully.
# CRITICAL: Model setup (downloading weights, configuring paths) can be time-consuming and complex. Provide clear, step-by-step instructions in the README.
# CRITICAL: Handling various video codecs and formats can be complex; ensure robust error handling for unsupported formats.
# CRITICAL: Large video files will result in long transcription times and high resource usage.
# CRITICAL: Topic extraction from raw transcripts can be noisy. The quality of extracted topics will depend on the NLP techniques used.
```

## Implementation Blueprint

### Data models and structure
(Not applicable for this spike, as it focuses on core functionality rather than data persistence.)

### list of tasks to be completed to fullfill the PRP in the order they should be completed

```yaml
Task 1: Setup Minimal Project Structure and CLI for Spike
CREATE src/main.py:
  - Setup basic CLI using `argparse` or `click`.
  - Define a top-level `spike-ingest` command.

CREATE src/video_ingestion/__init__.py:
CREATE src/cli/__init__.py:
CREATE src/cli/commands.py:
  - Define the `spike_ingest` command function.

Task 2: Implement Audio Extraction
CREATE src/video_ingestion/audio_extractor.py:
  - Function to extract audio from MP4 using `ffmpeg` and `pydub`.
  - Handle temporary audio file storage.

Task 3: Implement Image Extraction
CREATE src/video_ingestion/image_extractor.py:
  - Function to capture screenshots every 4 seconds using `ffmpeg`.
  - Generate timestamped filenames: `{video_name}_{timestamp}.jpg` (e.g., `test_video_004s.jpg`).
  - Handle output directory creation and cleanup.

Task 4: Implement Whisper Transcriber
CREATE src/video_ingestion/transcriber.py:
  - Implement `load_whisper_model()` to load the actual Whisper model (user will need to install it).
  - Implement `transcribe_audio()` to perform actual transcription using the loaded Whisper model.
  - This task will require the user to follow Whisper installation instructions.

Task 5: Implement Keyword and Timestamp Extraction
CREATE src/video_ingestion/topic_extractor.py:
  - Function to take a Whisper transcription result (including word-level timestamps).
  - Identify predefined keywords related to 'price action' and 'technical analysis'.
  - Extract each keyword along with its start and end timestamps.

Task 6: Orchestrate Spike Process and Markdown Output
MODIFY src/cli/commands.py:
  - Integrate `audio_extractor`, `image_extractor`, `transcriber`, and `topic_extractor` modules within the `spike_ingest` command.
  - Implement progress reporting and error handling.
  - Format the extracted keywords and timestamps into a readable markdown file.

Task 7: Environment Variable and Setup Instructions
CREATE .env.example:
  - Add clear, detailed instructions for `ffmpeg` installation and Whisper model download/setup.
```

### Per task pseudocode as needed added to each task
```python

# Task 2: Implement Audio Extraction
# src/video_ingestion/audio_extractor.py
import subprocess
from pydub import AudioSegment
import os

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

# Task 3: Implement Image Extraction
# src/video_ingestion/image_extractor.py
import subprocess
import os
from pathlib import Path

def extract_screenshots(video_path: str, output_dir: str, interval_seconds: int = 4) -> List[str]:
    """
    Extracts screenshots from an MP4 video file every N seconds using ffmpeg.
    Returns list of generated screenshot filenames.
    """
    video_name = Path(video_path).stem  # Get filename without extension
    os.makedirs(output_dir, exist_ok=True)
    
    # ffmpeg command to extract frames every N seconds
    # Format: video_name_004s.jpg, video_name_008s.jpg, etc.
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

# Task 4: Implement Whisper Transcriber
# src/video_ingestion/transcriber.py
import os
import torch
import whisper

# Global variables for model
_whisper_model = None

def load_whisper_model():
    """
    Loads the Whisper model.
    User MUST install openai-whisper.
    """
    global _whisper_model
    if _whisper_model is None:
        print("Loading Whisper model (base). This may take some time.")
        try:
            _whisper_model = whisper.load_model("base")
            print("Whisper model loaded successfully.")
        except Exception as e:
            raise RuntimeError(f"Failed to load Whisper model. Ensure it's installed and configured correctly: {e}")
    return _whisper_model

def transcribe_audio(audio_path: str) -> str:
    """
    Transcribes an audio file using the Whisper model.
    """
    model = load_whisper_model()
    print(f"Transcribing audio from {audio_path} using Whisper...")
    try:
        result = model.transcribe(audio_path)
        return result["text"]
    except Exception as e:
        raise RuntimeError(f"Whisper transcription failed: {e}")

# Task 4: Implement Keyword and Timestamp Extraction
# src/video_ingestion/topic_extractor.py
import spacy
from typing import List, Dict, Any

# Load a spaCy model (e.g., 'en_core_web_sm' or 'en_core_web_md')
# User might need to run: python -m spacy download en_core_web_sm
try:
    _nlp = spacy.load("en_core_web_sm")
except OSError:
    
    _nlp = spacy.load("en_core_web_sm")

def extract_topics_with_timestamps(whisper_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extracts specific keywords and their timestamps from a Whisper transcription result.
    Focuses on 'price action' and 'technical analysis' related terms.
    """
    keywords_of_interest = [
        "price", "action", "support", "resistance", "trend", "breakout",
        "candlestick", "chart", "indicator", "moving average", "RSI", "MACD",
        "volume", "pattern", "bearish", "bullish", "long", "short", "trade"
    ]

    extracted_keywords = []

    if "segments" in whisper_result:
        for segment in whisper_result["segments"]:
            if "words" in segment:
                for word_info in segment["words"]:
                    word = word_info["word"].lower().strip()
                    if word in keywords_of_interest:
                        extracted_keywords.append({
                            "keyword": word,
                            "start": word_info["start"],
                            "end": word_info["end"]
                        })
    return extracted_keywords

# Task 6: Orchestrate Spike Process and Markdown Output (Example in src/cli/commands.py)
# src/cli/commands.py
import argparse
import os
import tempfile
import uuid # For unique IDs
import datetime # For timestamps
from src.video_ingestion.audio_extractor import extract_audio
from src.video_ingestion.image_extractor import extract_screenshots
from src.video_ingestion.transcriber import transcribe_audio
from src.video_ingestion.topic_extractor import extract_topics_with_timestamps
from moviepy.editor import VideoFileClip # For getting video duration

def spike_ingest_command(args):
    """
    CLI command for the video ingestion spike.
    """
    video_path = args.video_path
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return

    # Get video duration
    try:
        clip = VideoFileClip(video_path)
        video_duration = clip.duration
        clip.close()
        print(f"Video duration: {video_duration:.2f} seconds")
    except Exception as e:
        print(f"Error getting video duration: {e}. Cannot proceed with timestamping.")
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        audio_output_path = os.path.join(tmpdir, "extracted_audio.wav")
        output_dir = os.path.join(os.getcwd(), "output")
        screenshots_dir = os.path.join(output_dir, "screenshots")
        reports_dir = os.path.join(output_dir, "reports")
        output_markdown_path = os.path.join(reports_dir, f"spike_output_{uuid.uuid4().hex[:8]}.md")
        
        # Create output directories
        os.makedirs(screenshots_dir, exist_ok=True)
        os.makedirs(reports_dir, exist_ok=True)

        try:
            extract_audio(video_path, audio_output_path)
            screenshot_files = extract_screenshots(video_path, screenshots_dir, interval_seconds=4)
            full_transcript = transcribe_audio(audio_output_path)
            
            topics_data = extract_topics_with_timestamps(full_transcript, video_duration)

            with open(output_markdown_path, "w") as f:
                f.write(f"# Video Ingestion Spike Report for: {os.path.basename(video_path)}\n\n")
                f.write(f"**Date:** {datetime.datetime.now().isoformat()}\n\n")
                
                f.write("## Extracted Screenshots\n\n")
                f.write(f"Screenshots captured every 4 seconds ({len(screenshot_files)} total):\n\n")
                for screenshot in screenshot_files:
                    rel_path = os.path.relpath(screenshot, reports_dir)
                    f.write(f"- `{rel_path}`\n")
                f.write("\n")
                
                f.write("## Full Transcript\n\n")
                f.write(full_transcript)
                f.write("\n\n## Identified Topics with Timestamps\n\n")
                for i, topic_info in enumerate(topics_data):
                    f.write(f"{i+1}. **Topic:** {topic_info['topic']}\n")
                    f.write(f"   **Time:** {topic_info['start_time']:.2f}s - {topic_info['end_time']:.2f}s\n")
                    f.write(f"   **Segment:** \"{topic_info['text_segment']}\"\n\n")
            
            print(f"Spike successful!")
            print(f"Report generated at: {output_markdown_path}")
            print(f"Screenshots saved to: {screenshots_dir}")
            print(f"All outputs saved to: {output_dir}")

        except RuntimeError as e:
            print(f"Spike failed: {e}")
        finally:
            # Clean up temporary audio file if it exists
            if os.path.exists(audio_output_path):
                os.remove(audio_output_path)

def add_spike_commands(parser):
    """Adds spike-specific commands to the CLI parser."""
    spike_parser = parser.add_parser('spike-ingest', help='Run an end-to-end spike for video ingestion technology validation.')
    spike_parser.add_argument('--video-path', required=True, help='Path to the MP4 video file for the spike.')
    spike_parser.set_defaults(func=spike_ingest_command)

# src/main.py (main entry point)
# import argparse
# from src.cli.commands import add_spike_commands

# def main():
#     parser = argparse.ArgumentParser(description="Video Analysis CLI")
#     subparsers = parser.add_subparsers(dest='command', help='Available commands')
#     add_spike_commands(subparsers) # Add spike commands
#     args = parser.parse_args()
#     if hasattr(args, 'func'):
#         args.func(args)
#     else:
#         parser.print_help()

# if __name__ == '__main__':
#     main()
```

### Integration Points
```yaml
CLI:
  - add to: src/main.py (CLI entry point)
  - add to: src/cli/commands.py (new `spike-ingest` command)
  - pattern: "parser.add_argument('--video-path', help='Path to the MP4 video file for the spike.')"

CONFIG:
  - add to: .env.example
  - pattern: "# Instructions for ffmpeg installation: https://ffmpeg.org/download.html"
  - pattern: "# Instructions for Whisper setup: Refer to https://github.com/openai/whisper"
  - pattern: "# Instructions for spaCy model download: python -m spacy download en_core_web_sm"
```

## Validation Loop

### Level 1: Syntax & Style
```bash
# Run these FIRST - fix any errors before proceeding
ruff check src/ --fix  # Auto-fix what's possible
mypy src/              # Type checking

# Expected: No errors. If errors, READ the error and fix.
```

### Level 2: Unit Tests each new feature/file/function use existing test patterns
```python
# CREATE tests/test_video_ingestion/test_audio_extractor.py
import pytest
import os
import subprocess
from src.video_ingestion.audio_extractor import extract_audio

def test_extract_audio_success(tmp_path):
    """Test successful audio extraction."""
    dummy_video_path = tmp_path / "dummy_video.mp4"
    # Create a very small, silent MP4 file using ffmpeg for testing
    subprocess.run([
        "ffmpeg", "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100,t=1",
        "-c:a", "aac", "-b:a", "128k", str(dummy_video_path)
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

# CREATE tests/test_video_ingestion/test_transcriber.py
import pytest
from unittest.mock import patch
from src.video_ingestion.transcriber import load_video_llama_model, transcribe_audio

@patch('src.video_ingestion.transcriber._video_llama_model', None)
@patch('src.video_ingestion.transcriber._video_llama_processor', None)
def test_load_video_llama_model_success():
    """Test successful (simulated) loading of Video-Llama model."""
    model, processor = load_video_llama_model()
    assert model is True
    assert processor is True

@patch('src.video_ingestion.transcriber.load_video_llama_model', return_value=(True, True))
def test_transcribe_audio_success(mock_load_model):
    """Test successful (simulated) transcription."""
    transcript = transcribe_audio("dummy_audio.wav")
    assert "transcription" in transcript and "Video-Llama" in transcript
    mock_load_model.assert_called_once()

@patch('src.video_ingestion.transcriber.load_video_llama_model', side_effect=Exception("Model init error"))
def test_transcribe_audio_model_load_failure(mock_load_model):
    """Test transcription when model loading fails."""
    with pytest.raises(RuntimeError, match="Video-Llama transcription failed: Model init error"):
        transcribe_audio("dummy_audio.wav")

# CREATE tests/test_video_ingestion/test_topic_extractor.py
import pytest
from src.video_ingestion.topic_extractor import extract_topics_with_timestamps

def test_extract_topics_with_timestamps_basic():
    """Test basic topic extraction and timestamping."""
    transcript = "This is a sentence about market analysis. Another sentence discusses trading strategies. Finally, we talk about risk management."
    video_duration = 60.0 # seconds
    topics = extract_topics_with_timestamps(transcript, video_duration)

    assert len(topics) == 3 # Expect 3 sentences/topics
    assert "market" in topics[0]["topic"]
    assert "trading" in topics[1]["topic"]
    assert "risk" in topics[2]["topic"]
    
    # Check timestamps are reasonable (not exact, but increasing)
    assert topics[0]["start_time"] == 0.0
    assert topics[0]["end_time"] > 0
    assert topics[1]["start_time"] == topics[0]["end_time"]
    assert topics[2]["start_time"] == topics[1]["end_time"]
    assert topics[-1]["end_time"] <= video_duration + 1 # Allow for slight floating point inaccuracies

def test_extract_topics_with_timestamps_empty_transcript():
    """Test with an empty transcript."""
    topics = extract_topics_with_timestamps("", 10.0)
    assert topics == []

def test_extract_topics_with_timestamps_zero_duration():
    """Test with zero video duration."""
    transcript = "Some text."
    topics = extract_topics_with_timestamps(transcript, 0.0)
    assert topics[0]["start_time"] == 0.0
    assert topics[0]["end_time"] == 0.0 # Should be 0 if duration is 0
```

```bash
# Run and iterate until passing:
uv run pytest tests/test_video_ingestion/ -v
# If failing: Read error, understand root cause, fix code, re-run (never mock to pass)
```

### Level 3: Integration Test
```bash
# Manual test:
# 1. Ensure you have a small MP4 video file (e.g., 10-30 seconds) for testing.
# 2. IMPORTANT: Install `openai-whisper` and ensure `torch` is installed. The Whisper model will download automatically on first use.
# 3. Ensure `ffmpeg` is installed and in your system's PATH.
# 4. Ensure spaCy model 'en_core_web_sm' is downloaded: `python -m spacy download en_core_web_sm`
# 5. Run the spike command:
#    uv run python src/main.py spike-ingest --video-path /path/to/your/test_video.mp4

# Expected Output:
# - Console messages indicating audio extraction progress.
# - Console messages indicating Whisper model loading and transcription progress.
# - A new markdown file (e.g., `spike_output_xxxx.md`) will be created in your current working directory.

# Expected Content of Markdown File:
# - A clear title with the video filename.
# - The full transcript generated by Whisper.
# - A structured list of identified keywords, each with estimated start and end timestamps.

# Expected Failures (and their messages):
# - If ffmpeg is not installed: "Spike failed: ffmpeg not found. Please install ffmpeg and ensure it's in your system's PATH."
# - If video path is invalid: "Error: Video file not found at /path/to/your/test_video.mp4"
# - If Whisper model loading fails (e.g., missing weights, incorrect path, insufficient GPU): "Spike failed: Failed to load Whisper model. Ensure it's installed and configured correctly: ..."
# - If Whisper transcription fails: "Spike failed: Whisper transcription failed: ..."
```

## Final validation Checklist
- [ ] All tests pass: `uv run pytest tests/ -v`
- [ ] No linting errors: `uv run ruff check src/`
- [ ] No type errors: `uv run mypy src/`
- [ ] Manual test successful: `uv run python src/main.py spike-ingest --video-path /path/to/your/test_video.mp4`
- [ ] Error cases (ffmpeg not found, invalid video path, Video-Llama issues) are handled gracefully and provide clear messages.
- [ ] Logs are informative but not verbose.
- [ ] `ffmpeg` is confirmed to be installed and working.
- [ ] Whisper model is confirmed to be loadable and performs actual transcription.
- [ ] The generated markdown file contains the full transcript and correctly formatted keywords with timestamps.

---

## Anti-Patterns to Avoid
- ❌ Don't skip `ffmpeg` or Whisper installation; this spike *requires* them to be functional.
- ❌ Don't hardcode paths; use `os.path.join` and temporary directories for intermediate files.
- ❌ Don't ignore Whisper model resource requirements (e.g., GPU, memory); clearly communicate these to the user.
- ❌ Don't expect perfect topic extraction or timestamping in this spike; the goal is basic functionality validation.