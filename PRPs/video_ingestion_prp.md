name: "Video Ingestion Feature"
description: |

## Purpose
This PRP outlines the requirements for the video ingestion feature, allowing users to process their local MP4 video files for transcription, topic extraction, and cataloging. This is a foundational step for the overall project, enabling the subsequent recommendation engine.

## Core Principles
1. **Context is King**: Include ALL necessary documentation, examples, and caveats
2. **Validation Loops**: Provide executable tests/lints the AI can run and fix
3. **Information Dense**: Use keywords and patterns from the codebase
4. **Progressive Success**: Start simple, validate, then enhance
5. **Global rules**: Be sure to follow all rules in GEMINI.md

---

## Goal
To develop a robust, user-driven video ingestion pipeline that takes local MP4 video files, extracts audio and screenshots, transcribes using the Whisper model, and identifies keywords with timestamps. The pipeline outputs structured reports and media files for further processing by the catalog system. The process should be resilient to common video issues and provide clear feedback to the user.

## Why
- This feature is critical for the product, as users must be able to process their own purchased video content.
- It provides the foundation for the catalog database system by generating structured data files.
- It establishes the core video processing pipeline that other features will build upon.
- It provides a clear entry point for users to utilize the software with their existing video library.

## What
The video ingestion feature will:
- Accept a local file path to an MP4 video as input.
- Validate the input video file (e.g., existence, format).
- Extract the audio track from the video.
- Capture screenshots every 4 seconds using ffmpeg, saving them with timestamped filenames that include the original video name and timestamp (e.g., `video_name_004s.jpg`, `video_name_008s.jpg`).
- Transcribe the audio using the Whisper model.
- Process the transcription using spaCy NLP to identify specific trading/financial keywords (e.g., price action, support, resistance, technical analysis terms) and their exact timestamps, returning a dictionary mapping each keyword to all its timestamp occurrences.
- Use lemmatization for enhanced matching (e.g., "trading" matches "trade", "levels" matches "level").
- Generate structured markdown reports containing the full transcript, identified keywords with timestamps, and references to captured screenshots.
- Organize outputs into structured directories (`output/reports/`, `output/screenshots/`) for better file management.
- Provide progress updates and error handling during the ingestion process.

### Success Criteria
- [ ] User can successfully ingest an MP4 video file via a CLI command.
- [ ] Audio is accurately extracted from the video.
- [ ] Screenshots are successfully captured every 4 seconds with timestamped filenames.
- [ ] Screenshot filenames follow the pattern: `{video_name}_{timestamp}.jpg` (e.g., `test_video_004s.jpg`).
- [ ] Transcription is successfully generated using the Whisper model.
- [ ] Specific keywords and their timestamps are successfully extracted from the transcription using spaCy NLP.
- [ ] Keywords are returned in dictionary format: `{keyword: [{"start": 1.2, "end": 1.5}, ...]}`.
- [ ] Lemmatization correctly matches word variations (e.g., "trading" → "trade").
- [ ] Structured markdown reports are generated with transcript, keywords, and screenshot references.
- [ ] Outputs are organized into structured directories (`output/reports/`, `output/screenshots/`).
- [ ] The system handles common errors gracefully (e.g., invalid file path, model loading errors).
- [ ] Progress is reported to the user during long-running operations.

## All Needed Context

### Documentation & References (list all context needed to implement the feature)
```yaml
# MUST READ - Include these in your context window
- url: https://github.com/openai/whisper
  why: Official GitHub repository for OpenAI Whisper, including installation and usage.

- url: https://github.com/openai/whisper#available-models-and-languages
  why: Whisper documentation on available models and their sizes.

- url: https://ffmpeg.org/documentation.html
  why: Official documentation for ffmpeg, specifically for audio extraction from video.

- url: https://pydub.com/
  why: Python library for audio manipulation, useful for handling extracted audio.

- url: https://docs.python.org/3/library/sqlite3.html
  why: Python's built-in SQLite documentation for local database storage.

- url: https://spacy.io/usage/linguistic-features
  why: spaCy documentation for NLP features including lemmatization and tokenization.

- url: https://spacy.io/usage/models
  why: spaCy model installation guide for en_core_web_sm model.

- file: GEMINI.md
  why: Project-specific guidelines and AI behavior rules.

- file: INITIAL.md
  why: Overall project overview and high-level requirements.

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
    └── templates/
        └── prp_base.md
```

### Desired Codebase tree with files to be added and responsibility of file
```bash
.
├── src/
│   ├── __init__.py
│   ├── main.py             # CLI entry point, orchestrates ingestion
│   ├── video_ingestion/
│   │   ├── __init__.py
│   │   ├── audio_extractor.py # Handles audio extraction using ffmpeg/pydub
│   │   ├── image_extractor.py # Handles screenshot extraction using ffmpeg
│   │   ├── transcriber.py  # Handles transcription using Whisper model
│   │   └── topic_extractor.py # Handles keyword extraction and timestamping
│   └── cli/
│       ├── __init__.py
│       └── commands.py     # Defines CLI commands for ingestion
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
# CRITICAL: Whisper models can be large and require significant computational resources (e.g., GPU). Inform users about hardware requirements.
# CRITICAL: Model setup (downloading weights, configuring paths) can be complex. Provide clear, step-by-step instructions.
# CRITICAL: ffmpeg must be installed and accessible in the system's PATH for audio extraction and screenshot capture. Provide clear instructions for users.
# CRITICAL: spaCy en_core_web_sm model must be installed: `python -m spacy download en_core_web_sm`. Handle OSError if model is missing.
# CRITICAL: Handling various video codecs and formats can be complex; ensure robust error handling for unsupported formats.
# CRITICAL: Large video files will result in long transcription times. Provide progress indicators for user feedback.
# CRITICAL: spaCy lemmatization may produce unexpected results (e.g., "levels" → "level"). Design keyword extraction to handle this appropriately.
# CRITICAL: Screenshot extraction timing must align with transcription timestamps for accurate report generation.
# CRITICAL: Markdown report generation should use relative paths for screenshot references to maintain portability.
```

## Implementation Blueprint

### list of tasks to be completed to fullfill the PRP in the order they should be completed

```yaml
Task 1: Setup Project Structure and Initial CLI
CREATE src/main.py:
  - Setup basic CLI using `argparse` or `click`.
  - Define a top-level `ingest` command.

CREATE src/video_ingestion/__init__.py:
CREATE src/cli/__init__.py:
CREATE src/cli/commands.py:
  - Define the `ingest` command function.

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
  - Function to transcribe audio using Whisper model.
  - Handle model loading and inference.
  - Return transcription with word-level timestamps when possible.
  - Handle potential errors during model execution.

Task 5: Implement Keyword Extraction with spaCy
CREATE src/video_ingestion/topic_extractor.py:
  - Function to extract trading/financial keywords from Whisper transcription with word-level timestamps.
  - Use spaCy NLP for enhanced keyword matching including lemmatization.
  - Return dictionary format: `{keyword: [{"start": 1.2, "end": 1.5}, ...]}`.
  - Handle punctuation cleaning and avoid duplicate timestamps.

Task 6: Orchestrate Ingestion Process and Report Generation
MODIFY src/cli/commands.py:
  - Integrate `audio_extractor`, `image_extractor`, `transcriber`, and `topic_extractor` modules within the `ingest` command.
  - Generate structured markdown reports with transcript, keywords, and screenshot references.
  - Implement progress reporting and error handling.
  - Ensure outputs are organized into structured directories (`output/reports/`, `output/screenshots/`).

Task 7: Environment Variable Setup
CREATE .env.example:
  - Add clear, detailed instructions for `ffmpeg` installation and Whisper model download/setup.
  - Add spaCy model installation instructions: `python -m spacy download en_core_web_sm`.

```

### Per task pseudocode as needed added to each task
```python

# Task 2: Implement Audio Extraction
# src/video_ingestion/audio_extractor.py
import subprocess
from pydub import AudioSegment

def extract_audio(video_path: str, output_audio_path: str) -> None:
    """
    Extracts audio from an MP4 video file using ffmpeg.
    """
    # CRITICAL: Ensure ffmpeg is installed and in PATH
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
        subprocess.run(command, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        # Reason: Provide clear error if ffmpeg fails
        raise RuntimeError(f"ffmpeg audio extraction failed: {e.stderr.decode()}")

# Task 3: Implement Image Extraction
# src/video_ingestion/image_extractor.py
import subprocess
import os
from pathlib import Path
from typing import List

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
        subprocess.run(command, check=True, capture_output=True, text=True)
        
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

```

### Integration Points
```yaml
CONFIG:
  - add to: .env.example
  - pattern: "# Instructions for ffmpeg installation: https://ffmpeg.org/download.html"
  - pattern: "# Instructions for Whisper setup: Refer to https://github.com/openai/whisper"
  - pattern: "# Instructions for spaCy model: python -m spacy download en_core_web_sm"

CLI:
  - add to: src/main.py (CLI commands)
  - pattern: "parser.add_argument('--video-path', help='Path to the MP4 video file to ingest.')"

OUTPUT:
  - directories: "output/reports/ and output/screenshots/ for organized file management"
  - format: "Structured markdown reports with relative paths to screenshots"
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
from src.video_ingestion.audio_extractor import extract_audio

def test_extract_audio_success(tmp_path):
    """Test successful audio extraction."""
    # Create a dummy video file for testing (requires ffmpeg)
    # This is a placeholder, actual test would need a small dummy video
    dummy_video_path = tmp_path / "dummy_video.mp4"
    # Create a very small, silent MP4 file using ffmpeg for testing
    # Reason: Need a valid MP4 for ffmpeg to process, even if it's minimal.
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

# CREATE tests/test_video_ingestion/test_image_extractor.py
import pytest
import os
import subprocess
from unittest.mock import patch
from src.video_ingestion.image_extractor import extract_screenshots

def test_extract_screenshots_success(tmp_path):
    """Test successful screenshot extraction."""
    dummy_video_path = tmp_path / "test_video.mp4"
    # Create a dummy video file using ffmpeg
    subprocess.run([
        "ffmpeg", "-f", "lavfi", "-i", "color=black:size=320x240:duration=8",
        "-c:v", "libx264", "-y", str(dummy_video_path)
    ], check=True, capture_output=True)
    
    output_dir = tmp_path / "screenshots"
    screenshot_files = extract_screenshots(str(dummy_video_path), str(output_dir), interval_seconds=4)
    
    assert len(screenshot_files) >= 1  # Should have at least one screenshot
    for screenshot_file in screenshot_files:
        assert os.path.exists(screenshot_file)
        assert screenshot_file.endswith('.jpg')
        assert 'test_video' in os.path.basename(screenshot_file)

def test_extract_screenshots_invalid_video():
    """Test screenshot extraction with invalid video file."""
    with pytest.raises(RuntimeError, match="ffmpeg screenshot extraction failed"):
        extract_screenshots("/nonexistent/video.mp4", "/tmp/screenshots")

@patch('subprocess.run')
def test_extract_screenshots_ffmpeg_not_found(mock_subprocess):
    """Test screenshot extraction when ffmpeg is not available."""
    mock_subprocess.side_effect = FileNotFoundError("ffmpeg not found")
    
    with pytest.raises(RuntimeError, match="ffmpeg not found"):
        extract_screenshots("video.mp4", "output_dir")

# CREATE tests/test_video_ingestion/test_transcriber.py
import pytest
import os
from unittest.mock import patch, MagicMock
from src.video_ingestion.transcriber import transcribe_audio, load_video_llama_model

@patch('src.video_ingestion.transcriber.model', None) # Ensure model is not pre-loaded
@patch('src.video_ingestion.transcriber.processor', None) # Ensure processor is not pre-loaded
@patch('src.video_ingestion.transcriber.load_video_llama_model') # Mock the loading function
def test_transcribe_audio_success(mock_load_model, tmp_path):
    """Test successful audio transcription with Video-Llama placeholder."""
    dummy_audio_path = tmp_path / "dummy_audio.mp3"
    dummy_audio_path.write_bytes(b"dummy audio data")

    transcript = transcribe_audio(str(dummy_audio_path))
    assert transcript == "This is a placeholder transcription from Video-Llama."
    mock_load_model.assert_called_once() # Ensure model loading was attempted

@patch('src.video_ingestion.transcriber.model', None)
@patch('src.video_ingestion.transcriber.processor', None)
@patch('src.video_ingestion.transcriber.load_video_llama_model', side_effect=Exception("Model load error"))
def test_transcribe_audio_model_load_failure(mock_load_model, tmp_path):
    """Test audio transcription when Video-Llama model loading fails."""
    dummy_audio_path = tmp_path / "dummy_audio.mp3"
    dummy_audio_path.write_bytes(b"dummy audio data")

    with pytest.raises(RuntimeError, match="Whisper transcription failed: Model load error"):
        transcribe_audio(str(dummy_audio_path))

```

```bash
# Run and iterate until passing:
uv run pytest tests/test_video_ingestion/ -v
# If failing: Read error, understand root cause, fix code, re-run (never mock to pass)
```

### Level 3: Integration Test
```bash
# Manual test:
# 1. Ensure ffmpeg is installed and in your system's PATH.
# 2. IMPORTANT: Install `openai-whisper` and ensure `torch` is installed. The Whisper model will download automatically on first use.
# 3. Ensure spaCy model is installed: `python -m spacy download en_core_web_sm`
# 4. Place a small MP4 video file (e.g., 30 seconds) in a known location.
# 5. Run the ingestion command:
#    uv run python src/main.py ingest --video-path /path/to/your/video.mp4

# Expected:
# - Progress messages in the console during processing
# - Structured directories created: output/reports/ and output/screenshots/
# - Markdown report generated in output/reports/ with transcript and keywords
# - Screenshots captured every 4 seconds in output/screenshots/
# - Report contains relative paths to screenshots for portability
# - No errors or exceptions are reported
```

## Final validation Checklist
- [ ] All tests pass: `uv run pytest tests/ -v`
- [ ] No linting errors: `uv run ruff check src/`
- [ ] No type errors: `uv run mypy src/`
- [ ] Manual test successful: `uv run python src/main.py ingest --video-path /path/to/test_video.mp4`
- [ ] Error cases handled gracefully (missing ffmpeg, invalid video paths, model loading failures)
- [ ] Progress messages are informative but not verbose
- [ ] ffmpeg is confirmed to be installed and working for both audio extraction and screenshot capture
- [ ] Whisper model is confirmed to be loadable and performs actual transcription
- [ ] spaCy model downloads and performs keyword extraction correctly
- [ ] Markdown reports are generated with proper structure and relative screenshot paths

## Anti-Patterns to Avoid
- ❌ Don't hardcode model paths or sensitive information - use environment variables
- ❌ Don't assume ffmpeg is installed; provide clear instructions and error messages
- ❌ Don't process entire large videos in memory; use streaming for audio extraction
- ❌ Don't ignore Whisper model resource requirements (e.g., GPU, memory)
- ❌ Don't use absolute paths in markdown reports - use relative paths for portability
- ❌ Don't skip progress reporting for long-running operations like transcription
- ❌ Don't create screenshots without proper timestamp alignment with transcription
