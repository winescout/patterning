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
To develop a robust, user-driven video ingestion pipeline that takes local MP4 video files, extracts audio, transcribes it using the Gemini API, identifies topics, and stores relevant metadata in a catalog for future use. The process should be resilient to common video issues and provide clear feedback to the user.

## Why
- This feature is critical for the product, as users must be able to process their own purchased video content.
- It enables the core functionality of the recommendation engine by creating a searchable catalog of video slices.
- It provides a clear entry point for users to utilize the software with their existing video library.

## What
The video ingestion feature will:
- Accept a local file path to an MP4 video as input.
- Validate the input video file (e.g., existence, format).
- Extract the audio track from the video.
- Transcribe the audio using the Gemini API.
    - Process the transcription to identify key topics and segment the video into logical slices, storing these as timestamps for deeplinking rather than creating new video clips.
    - Identify and flag newly discovered topics for user review and approval.
- Store video metadata (original path, unique ID, start/end timestamps of slices, topics, keywords) in a local database.
- Provide progress updates and error handling during the ingestion process.

### Success Criteria
- [ ] User can successfully ingest an MP4 video file via a CLI command.
- [ ] Audio is accurately extracted from the video.
- [ ] Transcription is successfully generated using the Gemini API.
- [ ] Video content is segmented into logical topics.
- [ ] All relevant metadata for video slices is stored in the database.
- [ ] The system handles common errors gracefully (e.g., invalid file path, API errors).
- [ ] Progress is reported to the user during long-running operations.

## All Needed Context

### Documentation & References (list all context needed to implement the feature)
```yaml
# MUST READ - Include these in your context window
- url: https://ai.google.dev/docs/gemini_api_overview
  why: Overview of the Gemini API, authentication, and usage.

- url: https://ai.google.dev/docs/models/gemini
  why: Details on Gemini models, capabilities, and pricing for transcription.

- url: https://ffmpeg.org/documentation.html
  why: Official documentation for ffmpeg, specifically for audio extraction from video.

- url: https://pydub.com/
  why: Python library for audio manipulation, useful for handling extracted audio.

- url: https://docs.python.org/3/library/sqlite3.html
  why: Python's built-in SQLite documentation for local database storage.

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
│   │   ├── models.py       # Pydantic models for video/slice metadata
│   │   ├── audio_extractor.py # Handles audio extraction using ffmpeg/pydub
│   │   ├── transcriber.py  # Handles transcription using Gemini API
│   │   ├── topic_extractor.py # Handles topic modeling and segmentation
│   │   └── database.py     # Handles SQLite database operations for metadata
│   └── cli/
│       ├── __init__.py
│       └── commands.py     # Defines CLI commands for ingestion
├── tests/
│   ├── __init__.py
│   ├── test_video_ingestion/
│   │   ├── __init__.py
│   │   ├── test_audio_extractor.py
│   │   ├── test_transcriber.py
│   │   ├── test_topic_extractor.py
│   │   └── test_database.py
├── .env.example            # Example for API keys and other environment variables
├── database/
│   └── video_catalog.db    # SQLite database file (or similar)
```

### Known Gotchas of our codebase & Library Quirks
```python
# CRITICAL: Gemini API has rate limits and associated costs. Implement retry mechanisms and inform the user about potential costs.
# CRITICAL: ffmpeg must be installed and accessible in the system's PATH for audio extraction. Provide clear instructions for users.
# CRITICAL: Handling various video codecs and formats can be complex; ensure robust error handling for unsupported formats.
# CRITICAL: Large video files will result in long transcription times and potentially high API costs. Consider chunking or progress indicators.
# CRITICAL: Ensure proper handling of API keys (e.g., via environment variables using python_dotenv).
```

## Implementation Blueprint

### Data models and structure

Create the core data models, we ensure type safety and consistency.
```python
# src/video_ingestion/models.py

from pydantic import BaseModel
from typing import List, Dict, Optional

class VideoMetadata(BaseModel):
    """Metadata for an ingested video."""
    video_id: str  # Unique ID for the video
    original_path: str # Original file path provided by the user
    file_hash: str # Hash of the video file to detect changes/duplicates
    ingestion_timestamp: str # Timestamp of when the video was ingested

class VideoSlice(BaseModel):
    """Metadata for a segmented video slice."""
    slice_id: str # Unique ID for the slice
    video_id: str # Foreign key to VideoMetadata
    start_time: float # Start time in seconds
    end_time: float # End time in seconds
    transcript: str # Full transcript of the slice
    topics: List[str] # List of identified topics
    keywords: List[str] # List of extracted keywords
    summary: Optional[str] # Optional summary of the slice

class IngestionStatus(BaseModel):
    """Status of a video ingestion process."""
    video_id: str
    status: str # e.g., "PENDING", "AUDIO_EXTRACTED", "TRANSCRIBED", "COMPLETED", "FAILED"
    progress: float # 0.0 to 1.0
    message: Optional[str] # Detailed message or error
```

### list of tasks to be completed to fullfill the PRP in the order they should be completed

```yaml
Task 1: Setup Project Structure and Initial CLI
CREATE src/main.py:
  - Setup basic CLI using `argparse` or `click`.
  - Define a top-level `ingest` command.

CREATE src/video_ingestion/__init__.py:
CREATE src/video_ingestion/models.py:
  - Implement Pydantic models as defined in "Data models and structure".

CREATE src/cli/__init__.py:
CREATE src/cli/commands.py:
  - Define the `ingest` command function.

Task 2: Implement Audio Extraction
CREATE src/video_ingestion/audio_extractor.py:
  - Function to extract audio from MP4 using `ffmpeg` and `pydub`.
  - Handle temporary audio file storage.

Task 3: Implement Gemini Transcriber
CREATE src/video_ingestion/transcriber.py:
  - Function to send audio to Gemini API for transcription.
  - Handle API key loading from environment variables.
  - Implement retry logic for API calls.
  - Handle API rate limits and errors.

Task 4: Implement Database Module
CREATE src/video_ingestion/database.py:
  - Functions to initialize SQLite database (e.g., `video_catalog.db`).
  - Functions to store and retrieve `VideoMetadata` and `VideoSlice` objects.
  - Ensure proper indexing for efficient lookups.

Task 5: Implement Topic Extraction and Segmentation
CREATE src/video_ingestion/topic_extractor.py:
  - Function to take a full transcript and segment it into logical topics.
  - Use NLP techniques (e.g., sentence tokenization, keyword extraction, simple topic modeling).
  - Generate `VideoSlice` objects.

Task 6: Orchestrate Ingestion Process
MODIFY src/cli/commands.py:
  - Integrate `audio_extractor`, `transcriber`, `topic_extractor`, and `database` modules within the `ingest` command.
  - Implement progress reporting and error handling.
  - Update `VideoMetadata` and `VideoSlice` in the database.

Task 7: Environment Variable Setup
CREATE .env.example:
  - Add placeholder for `GEMINI_API_KEY`.
  - Add instructions for `ffmpeg` installation.

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

# Task 3: Implement Gemini Transcriber
# src/video_ingestion/transcriber.py
import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv() # Reason: Load environment variables for API key

def transcribe_audio(audio_path: str) -> str:
    """
    Transcribes an audio file using the Gemini API.
    """
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel('gemini-pro') # Reason: gemini-pro is good for text generation

    with open(audio_path, "rb") as audio_file:
        audio_content = audio_file.read()

    try:
        # CRITICAL: Handle potential API errors, rate limits, and cost implications
        response = model.generate_content(
            contents=[
                {"mime_type": "audio/mpeg", "data": audio_content} # Assuming mp3 for simplicity, adjust as needed
            ]
        )
        return response.text
    except Exception as e:
        # Reason: Catch broad exceptions for API failures
        raise RuntimeError(f"Gemini API transcription failed: {e}")

# Task 4: Implement Database Module
# src/video_ingestion/database.py
import sqlite3
from typing import List, Dict

def init_db(db_path: str):
    """Initializes the SQLite database schema."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS videos (
            video_id TEXT PRIMARY KEY,
            original_path TEXT NOT NULL,
            file_hash TEXT NOT NULL,
            ingestion_timestamp TEXT NOT NULL
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS video_slices (
            slice_id TEXT PRIMARY KEY,
            video_id TEXT NOT NULL,
            start_time REAL NOT NULL,
            end_time REAL NOT NULL,
            transcript TEXT NOT NULL,
            topics TEXT, -- Stored as JSON string
            keywords TEXT, -- Stored as JSON string
            summary TEXT,
            FOREIGN KEY (video_id) REFERENCES videos (video_id)
        )
    """)
    conn.commit()
    conn.close()

def insert_video_metadata(db_path: str, metadata: Dict):
    """Inserts video metadata into the database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO videos (video_id, original_path, file_hash, ingestion_timestamp)
        VALUES (?, ?, ?, ?)
    """, (metadata["video_id"], metadata["original_path"], metadata["file_hash"], metadata["ingestion_timestamp"]))
    conn.commit()
    conn.close()

def insert_video_slice(db_path: str, slice_data: Dict):
    """Inserts a video slice into the database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO video_slices (slice_id, video_id, start_time, end_time, transcript, topics, keywords, summary)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        slice_data["slice_id"], slice_data["video_id"], slice_data["start_time"],
        slice_data["end_time"], slice_data["transcript"],
        str(slice_data.get("topics", [])), # Reason: Store list as JSON string
        str(slice_data.get("keywords", [])), # Reason: Store list as JSON string
        slice_data.get("summary")
    ))
    conn.commit()
    conn.close()

```

### Integration Points
```yaml
DATABASE:
  - file: database/video_catalog.db
  - migration: "Initial schema creation for videos and video_slices tables."

CONFIG:
  - add to: .env.example
  - pattern: "GEMINI_API_KEY=YOUR_GEMINI_API_KEY"

ROUTES:
  - add to: src/main.py (CLI commands)
  - pattern: "parser.add_argument('--video-path', help='Path to the MP4 video file to ingest.')"
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

# CREATE tests/test_video_ingestion/test_transcriber.py
import pytest
import os
from unittest.mock import patch, MagicMock
from src.video_ingestion.transcriber import transcribe_audio

@patch('google.generativeai.GenerativeModel')
@patch('os.getenv', return_value='dummy_api_key')
def test_transcribe_audio_success(mock_getenv, mock_generative_model, tmp_path):
    """Test successful audio transcription."""
    mock_instance = mock_generative_model.return_value
    mock_instance.generate_content.return_value.text = "This is a test transcription."

    dummy_audio_path = tmp_path / "dummy_audio.mp3"
    dummy_audio_path.write_bytes(b"dummy audio data")

    transcript = transcribe_audio(str(dummy_audio_path))
    assert transcript == "This is a test transcription."
    mock_instance.generate_content.assert_called_once()

@patch('google.generativeai.GenerativeModel')
@patch('os.getenv', return_value='dummy_api_key')
def test_transcribe_audio_api_failure(mock_getenv, mock_generative_model, tmp_path):
    """Test audio transcription when Gemini API fails."""
    mock_instance = mock_generative_model.return_value
    mock_instance.generate_content.side_effect = Exception("API error")

    dummy_audio_path = tmp_path / "dummy_audio.mp3"
    dummy_audio_path.write_bytes(b"dummy audio data")

    with pytest.raises(RuntimeError, match="Gemini API transcription failed: API error"):
        transcribe_audio(str(dummy_audio_path))

# CREATE tests/test_video_ingestion/test_database.py
import pytest
import sqlite3
import os
from src.video_ingestion.database import init_db, insert_video_metadata, insert_video_slice
from src.video_ingestion.models import VideoMetadata, VideoSlice
import json

def test_init_db(tmp_path):
    """Test database initialization."""
    db_path = tmp_path / "test_catalog.db"
    init_db(str(db_path))
    assert os.path.exists(db_path)
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    conn.close()
    assert ('videos',) in tables
    assert ('video_slices',) in tables

def test_insert_video_metadata(tmp_path):
    """Test inserting video metadata."""
    db_path = tmp_path / "test_catalog.db"
    init_db(str(db_path))
    metadata = VideoMetadata(
        video_id="vid123",
        original_path="/path/to/video.mp4",
        file_hash="abc123xyz",
        ingestion_timestamp="2025-07-05T10:00:00Z"
    )
    insert_video_metadata(str(db_path), metadata.model_dump())

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM videos WHERE video_id='vid123'")
    result = cursor.fetchone()
    conn.close()
    assert result is not None
    assert result[0] == "vid123"

def test_insert_video_slice(tmp_path):
    """Test inserting a video slice."""
    db_path = tmp_path / "test_catalog.db"
    init_db(str(db_path))
    # First insert parent video metadata
    metadata = VideoMetadata(
        video_id="vid456",
        original_path="/path/to/video2.mp4",
        file_hash="def456uvw",
        ingestion_timestamp="2025-07-05T11:00:00Z"
    )
    insert_video_metadata(str(db_path), metadata.model_dump())

    video_slice = VideoSlice(
        slice_id="sliceA",
        video_id="vid456",
        start_time=10.5,
        end_time=30.0,
        transcript="This is a test transcription.",
        topics=["market open", "strategy"],
        keywords=["open", "strategy", "trade"],
        summary="Summary of market open strategy."
    )
    insert_video_slice(str(db_path), video_slice.model_dump())

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM video_slices WHERE slice_id='sliceA'")
    result = cursor.fetchone()
    conn.close()
    assert result is not None
    assert result[0] == "sliceA"
    assert result[1] == "vid456"
    assert json.loads(result[5].replace("'", "\"")) == ["market open", "strategy"] # Reason: Convert back from string to list for assertion

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
# 2. Create a .env file in the project root with your GEMINI_API_KEY.
#    Example: GEMINI_API_KEY=YOUR_API_KEY_HERE
# 3. Place a small MP4 video file (e.g., 30 seconds) in a known location.
# 4. Run the ingestion command:
#    uv run python src/main.py ingest --video-path /path/to/your/video.mp4

# Expected:
# - Progress messages in the console.
# - A 'database/video_catalog.db' file is created/updated.
# - The database contains entries in 'videos' and 'video_slices' tables for the ingested video.
# - No errors or exceptions are reported.
```

## Final validation Checklist
- [ ] All tests pass: `uv run pytest tests/ -v`
- [ ] No linting errors: `uv run ruff check src/`
- [ ] No type errors: `uv run mypy src/`
- [ ] Manual test successful: [specific curl/command]
- [ ] Error cases handled gracefully
- [ ] Logs are informative but not verbose
- [ ] Documentation updated if needed

---

## Anti-Patterns to Avoid
- ❌ Don't hardcode API keys or sensitive information.
- ❌ Don't assume ffmpeg is installed; provide clear instructions and error messages.
- ❌ Don't process entire large videos in memory; stream or chunk as necessary.
- ❌ Don't ignore Gemini API rate limits; implement backoff and retry.
- ❌ Don't store lists/dictionaries directly in SQLite without serialization (e.g., JSON).
