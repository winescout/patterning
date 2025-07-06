name: "Video Catalog Database System"
description: |

## Purpose
This PRP outlines the requirements for a comprehensive video catalog database system that stores and manages processed video metadata, enabling searchable video slice discovery and recommendation functionality. This system builds upon the video ingestion pipeline to create a persistent, queryable catalog of video content.

## Core Principles
1. **Context is King**: Include ALL necessary documentation, examples, and caveats
2. **Validation Loops**: Provide executable tests/lints the AI can run and fix
3. **Information Dense**: Use keywords and patterns from the codebase
4. **Progressive Success**: Start simple, validate, then enhance
5. **Global rules**: Be sure to follow all rules in CLAUDE.md

---

## Goal
To develop a robust SQLite-based catalog system that stores video metadata, transcription slices, keywords, and timestamps in a searchable format. The system should enable efficient querying for video recommendations, duplicate detection, and content discovery while maintaining data integrity and performance.

## Why
- **Enables the recommendation engine**: Creates the foundational data layer needed for video slice recommendations
- **Provides searchable catalog**: Users can discover relevant content across their entire video library
- **Prevents duplicate processing**: File hashing and metadata tracking avoid reprocessing the same content
- **Supports content analytics**: Enables analysis of keyword trends, content patterns, and user preferences
- **Scalable foundation**: Database design supports future features like user preferences, ratings, and advanced search

## What
The video catalog database system will:
- Design and implement a normalized SQLite database schema for video catalog data
- Create Pydantic models for type-safe data validation and serialization
- Implement database operations for storing, retrieving, and querying video metadata
- Provide video slice segmentation logic for breaking transcripts into meaningful chunks
- Enable efficient keyword and timestamp-based searches across the catalog
- Include file hashing for duplicate detection and change tracking
- Support progress tracking during ingestion with detailed status reporting
- Create database indexes for optimal query performance
- Implement data validation and integrity constraints
- Provide migration support for schema evolution

### Success Criteria
- [ ] SQLite database schema is properly normalized and indexed
- [ ] Pydantic models provide complete type safety for all data operations
- [ ] Video metadata can be stored and retrieved efficiently
- [ ] Video slices are segmented logically based on content boundaries
- [ ] Keyword searches return relevant video slices with timestamps
- [ ] File hashing prevents duplicate video processing
- [ ] Progress tracking accurately reflects ingestion status
- [ ] Database operations handle errors gracefully with proper rollbacks
- [ ] Query performance is optimized for typical search patterns
- [ ] Data integrity is maintained across all operations

## All Needed Context

### Documentation & References (list all context needed to implement the feature)
```yaml
# MUST READ - Include these in your context window
- url: https://docs.python.org/3/library/sqlite3.html
  why: Official SQLite documentation for Python, including transaction handling and best practices

- url: https://docs.pydantic.dev/latest/
  why: Pydantic v2 documentation for data validation, serialization, and model configuration

- url: https://docs.pydantic.dev/latest/concepts/validators/
  why: Custom validators for complex data validation (file hashes, timestamps, etc.)

- url: https://sqlite.org/lang_createindex.html
  why: SQLite index creation and optimization strategies for query performance

- url: https://sqlite.org/foreignkeys.html
  why: Foreign key constraints and referential integrity in SQLite

- url: https://sqlite.org/transactionmode.html
  why: Transaction modes and ACID compliance for data integrity

- file: src/video_ingestion/topic_extractor.py
  why: Existing keyword extraction patterns and data formats to integrate with

- file: src/video_ingestion/transcriber.py
  why: Understanding transcription output format for slice segmentation

- file: CLAUDE.md
  why: Project-specific guidelines and testing requirements

- file: PRPs/video_ingestion_prp.md
  why: Integration points with existing video processing pipeline
```

### Current Codebase tree (run `tree` in the root of the project) to get an overview of the codebase
```bash
.
├── .gitattributes
├── .gitignore
├── CLAUDE.md
├── INITIAL.md
├── LICENSE
├── README.md
├── .ai_assistant/
│   ├── settings.local.json
│   └── commands/
│       ├── execute-prp.md
│       └── generate-prp.md
├── src/
│   ├── __init__.py
│   ├── main.py
│   ├── video_ingestion/
│   │   ├── __init__.py
│   │   ├── audio_extractor.py
│   │   ├── image_extractor.py
│   │   ├── transcriber.py
│   │   └── topic_extractor.py
│   └── cli/
│       ├── __init__.py
│       └── commands.py
├── tests/
│   ├── __init__.py
│   ├── test_video_ingestion/
│   │   ├── __init__.py
│   │   ├── test_audio_extractor.py
│   │   ├── test_image_extractor.py
│   │   ├── test_transcriber.py
│   │   └── test_topic_extractor.py
│   ├── test_cli/
│   │   ├── __init__.py
│   │   └── test_commands.py
│   └── test_integration/
│       ├── __init__.py
│       └── test_spike_e2e.py
├── output/
│   ├── screenshots/
│   └── reports/
├── PRPs/
│   ├── video_ingestion_prp.md
│   ├── video_ingestion_spike_e2e_prp.md
│   ├── video_clip_clustering_prp.md
│   └── templates/
│       └── prp_base.md
└── spike_output_ac861f7b.md
```

### Desired Codebase tree with files to be added and responsibility of file
```bash
.
├── src/
│   ├── catalog/
│   │   ├── __init__.py
│   │   ├── models.py           # Pydantic models for catalog data
│   │   ├── database.py         # SQLite database operations and connections
│   │   ├── schema.py           # Database schema definition and migrations
│   │   ├── queries.py          # Complex queries and search operations
│   │   ├── video_slicer.py     # Logic for segmenting videos into slices
│   │   └── hash_utils.py       # File hashing utilities for duplicate detection
│   └── cli/
│       └── commands.py         # Updated to include catalog operations
├── tests/
│   ├── test_catalog/
│   │   ├── __init__.py
│   │   ├── test_models.py      # Test Pydantic models and validation
│   │   ├── test_database.py    # Test database operations
│   │   ├── test_schema.py      # Test schema creation and migrations
│   │   ├── test_queries.py     # Test search and query functions
│   │   ├── test_video_slicer.py # Test video segmentation logic
│   │   └── test_hash_utils.py  # Test file hashing functions
├── database/
│   ├── migrations/             # SQL migration files
│   │   └── 001_initial_schema.sql
│   └── video_catalog.db        # SQLite database file (created at runtime)
├── .env.example                # Updated with database configuration
```

### Known Gotchas of our codebase & Library Quirks
```python
# CRITICAL: SQLite requires proper transaction handling for data integrity
# Example: Always use context managers or explicit commit/rollback

# CRITICAL: Pydantic v2 has different syntax than v1 for validators and serialization
# Use @field_validator instead of @validator, model_dump() instead of dict()

# CRITICAL: SQLite foreign keys are disabled by default - must enable explicitly
# Execute: PRAGMA foreign_keys = ON; after connecting

# CRITICAL: Video slicing strategy must align with keyword timestamp formats
# Current format: {keyword: [{"start": 1.2, "end": 1.5}]} from topic_extractor.py

# CRITICAL: File hashing should use chunks for large files to avoid memory issues
# Don't load entire video files into memory for hashing

# CRITICAL: Database schema should support future features without breaking changes
# Design with extensibility in mind (user preferences, ratings, etc.)

# CRITICAL: spaCy keyword extraction uses lemmatization - database should store both forms
# Store original word and lemmatized form for comprehensive search
```

## Implementation Blueprint

### Data models and structure

Create the core data models to ensure type safety and consistency.
```python
# src/catalog/models.py

from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Optional, Any
from datetime import datetime
from enum import Enum

class IngestionStatus(str, Enum):
    """Status values for video ingestion process."""
    PENDING = "pending"
    AUDIO_EXTRACTED = "audio_extracted"
    SCREENSHOTS_CAPTURED = "screenshots_captured"
    TRANSCRIBED = "transcribed"
    KEYWORDS_EXTRACTED = "keywords_extracted"
    SLICED = "sliced"
    COMPLETED = "completed"
    FAILED = "failed"

class VideoMetadata(BaseModel):
    """Metadata for an ingested video."""
    video_id: str = Field(..., description="Unique identifier for the video")
    original_path: str = Field(..., description="Original file path provided by user")
    file_hash: str = Field(..., description="SHA256 hash of video file")
    file_size: int = Field(..., description="File size in bytes")
    duration: float = Field(..., description="Video duration in seconds")
    ingestion_timestamp: datetime = Field(default_factory=datetime.utcnow)
    status: IngestionStatus = Field(default=IngestionStatus.PENDING)
    error_message: Optional[str] = Field(default=None, description="Error details if status is FAILED")
    
    @field_validator('file_hash')
    @classmethod
    def validate_hash(cls, v: str) -> str:
        if len(v) != 64:  # SHA256 produces 64-character hex string
            raise ValueError('file_hash must be a valid SHA256 hash')
        return v.lower()

class VideoSlice(BaseModel):
    """Metadata for a segmented video slice."""
    slice_id: str = Field(..., description="Unique identifier for the slice")
    video_id: str = Field(..., description="Foreign key to VideoMetadata")
    start_time: float = Field(..., description="Start time in seconds")
    end_time: float = Field(..., description="End time in seconds") 
    transcript: str = Field(..., description="Full transcript of the slice")
    keywords: Dict[str, List[Dict[str, float]]] = Field(
        default_factory=dict, 
        description="Keywords with timestamps: {keyword: [{'start': 1.2, 'end': 1.5}]}"
    )
    
    @field_validator('start_time', 'end_time')
    @classmethod
    def validate_times(cls, v: float) -> float:
        if v < 0:
            raise ValueError('Time values must be non-negative')
        return v

class KeywordOccurrence(BaseModel):
    """Individual keyword occurrence with metadata."""
    keyword_id: str = Field(..., description="Unique identifier for this occurrence")
    slice_id: str = Field(..., description="Foreign key to VideoSlice")
    keyword: str = Field(..., description="Original keyword as found")
    lemmatized: str = Field(..., description="Lemmatized form of keyword")
    start_time: float = Field(..., description="Start time of keyword occurrence")
    end_time: float = Field(..., description="End time of keyword occurrence")
    confidence: Optional[float] = Field(default=None, description="Confidence score if available")

class SearchQuery(BaseModel):
    """Model for video search requests."""
    keywords: List[str] = Field(..., description="Keywords to search for")
    duration_min: Optional[float] = Field(default=None, description="Minimum slice duration")
    duration_max: Optional[float] = Field(default=None, description="Maximum slice duration")
    limit: int = Field(default=50, description="Maximum results to return")
```

### list of tasks to be completed to fulfill the PRP in the order they should be completed

```yaml
Task 1: Create Database Schema and Migration System
CREATE src/catalog/schema.py:
  - Design normalized SQLite schema with proper relationships
  - Create tables for videos, video_slices, keywords, and keyword_occurrences
  - Include proper indexes for query optimization
  - Enable foreign key constraints

CREATE database/migrations/001_initial_schema.sql:
  - SQL script for initial database creation
  - Indexes for common query patterns

Task 2: Implement Core Database Operations
CREATE src/catalog/database.py:
  - Database connection management with proper configuration
  - CRUD operations for all model types
  - Transaction handling with rollback support
  - Connection pooling for concurrent access

Task 3: Implement File Hashing Utilities
CREATE src/catalog/hash_utils.py:
  - Efficient file hashing for large video files
  - Duplicate detection logic
  - Change detection for file modifications

Task 4: Implement Video Slicing Logic  
CREATE src/catalog/video_slicer.py:
  - Algorithm to segment transcripts into logical slices
  - Integration with keyword timestamp data
  - Configurable slice duration and overlap parameters

Task 5: Implement Search and Query Operations
CREATE src/catalog/queries.py:
  - Keyword-based search across video catalog
  - Time-range and duration filtering
  - Relevance scoring for search results
  - Efficient pagination for large result sets

Task 6: Create Comprehensive Pydantic Models
MODIFY src/catalog/models.py:
  - Add remaining model validations and serialization
  - Include database relationship helpers
  - Add export/import functionality for catalog data

Task 7: Integrate with Video Ingestion Pipeline
MODIFY src/cli/commands.py:
  - Update ingestion command to use catalog database
  - Add progress tracking with status updates
  - Include error handling and rollback logic

Task 8: Create Database Management CLI Commands
ADD to src/cli/commands.py:
  - Database initialization and migration commands
  - Catalog search and query commands  
  - Database maintenance and cleanup operations
```

### Per task pseudocode as needed added to each task

```python
# Task 1: Database Schema Design
# src/catalog/schema.py
def create_database_schema(db_path: str):
    """Create complete database schema with relationships and indexes."""
    # PATTERN: Use context manager for connection handling
    with sqlite3.connect(db_path) as conn:
        # CRITICAL: Enable foreign keys for referential integrity
        conn.execute("PRAGMA foreign_keys = ON")
        
        # Create videos table with comprehensive metadata
        # GOTCHA: Include file_hash for duplicate detection
        # GOTCHA: Store both original and normalized paths
        
        # Create video_slices table with time boundaries
        # PATTERN: Use REAL type for precise timestamp storage
        # INDEX: Create indexes on video_id, start_time, end_time
        
        # Create keywords table for normalized keyword storage
        # PATTERN: Store both original and lemmatized forms
        
        # Create keyword_occurrences junction table
        # FOREIGN KEYS: Reference video_slices and keywords tables
        # INDEX: Optimize for keyword search queries

# Task 2: Database Operations
# src/catalog/database.py
class CatalogDatabase:
    """Main database interface with transaction support."""
    
    def __init__(self, db_path: str):
        # PATTERN: Use connection pooling for concurrent access
        self.db_path = db_path
        
    @contextmanager
    def get_connection(self):
        """Context manager for database connections."""
        # CRITICAL: Enable foreign keys on every connection
        # PATTERN: Auto-commit on success, rollback on exception
        
    async def store_video_metadata(self, metadata: VideoMetadata) -> bool:
        """Store video metadata with duplicate detection."""
        # GOTCHA: Check file_hash for duplicates before inserting
        # PATTERN: Use INSERT OR IGNORE for idempotent operations
        # CRITICAL: Update ingestion status atomically
        
    async def create_video_slices(self, slices: List[VideoSlice]) -> List[str]:
        """Batch insert video slices with keyword relationships."""
        # PATTERN: Use executemany for efficient batch operations
        # CRITICAL: Maintain transaction boundaries across related inserts
        # GOTCHA: Handle keyword normalization and deduplication

# Task 3: File Hashing
# src/catalog/hash_utils.py
def calculate_file_hash(file_path: str, chunk_size: int = 8192) -> str:
    """Calculate SHA256 hash of large files efficiently."""
    # CRITICAL: Read file in chunks to avoid memory issues
    # PATTERN: Use hashlib for consistent hashing
    # GOTCHA: Handle file access errors gracefully
    
def detect_duplicate_videos(db: CatalogDatabase, file_hash: str) -> Optional[VideoMetadata]:
    """Check if video already exists in catalog."""
    # PATTERN: Query by hash index for fast lookup
    # RETURN: Existing metadata if duplicate found

# Task 4: Video Slicing
# src/catalog/video_slicer.py  
def create_video_slices(
    transcript: str, 
    keywords_data: Dict[str, List[Dict[str, float]]],
    video_duration: float,
    slice_duration: float = 30.0
) -> List[VideoSlice]:
    """Segment transcript into logical slices with keyword alignment."""
    # ALGORITHM: Use keyword boundaries to create natural slice points
    # PATTERN: Ensure slices don't exceed max duration
    # GOTCHA: Handle edge cases where keywords span slice boundaries
    # OPTIMIZATION: Minimize slice overlap while preserving context

# Task 5: Search Queries
# src/catalog/queries.py
async def search_videos_by_keywords(
    db: CatalogDatabase, 
    query: SearchQuery
) -> List[VideoSlice]:
    """Search catalog using keyword matching with relevance scoring."""
    # PATTERN: Use JOIN queries for efficient keyword lookup
    # ALGORITHM: Score results by keyword frequency and co-occurrence
    # OPTIMIZATION: Use prepared statements for complex queries
    # GOTCHA: Handle lemmatized vs original keyword matching
```

### Integration Points
```yaml
DATABASE:
  - file: database/video_catalog.db
  - migration: "Initial schema with videos, slices, keywords, and occurrences tables"
  - indexes: "Optimize for keyword search, time range queries, and video lookup"

CONFIG:
  - add to: .env.example
  - pattern: "DATABASE_PATH=database/video_catalog.db"
  - pattern: "SLICE_DURATION=30.0  # Default slice length in seconds"

CLI:
  - add to: src/cli/commands.py
  - pattern: "parser.add_argument('--search', help='Search catalog by keywords')"
  - integration: "Replace markdown output with database storage in ingest command"

VIDEO_INGESTION:
  - modify: src/cli/commands.py spike_ingest_command
  - replace: Markdown file generation with database storage
  - add: Progress tracking and status updates during ingestion
```

## Validation Loop

### Level 1: Syntax & Style
```bash
# Run these FIRST - fix any errors before proceeding
ruff check src/catalog/ --fix  # Auto-fix what's possible
mypy src/catalog/              # Type checking

# Expected: No errors. If errors, READ the error and fix.
```

### Level 2: Unit Tests each new feature/file/function use existing test patterns
```python
# CREATE tests/test_catalog/test_models.py
def test_video_metadata_validation():
    """Test VideoMetadata model validation."""
    # Test valid metadata creation
    metadata = VideoMetadata(
        video_id="test-123",
        original_path="/path/to/video.mp4", 
        file_hash="a" * 64,  # Valid SHA256 hash
        file_size=1024000,
        duration=120.5
    )
    assert metadata.status == IngestionStatus.PENDING
    
    # Test invalid hash validation
    with pytest.raises(ValidationError):
        VideoMetadata(
            video_id="test-123",
            original_path="/path/to/video.mp4",
            file_hash="invalid_hash",  # Too short
            file_size=1024000,
            duration=120.5
        )

# CREATE tests/test_catalog/test_database.py  
@pytest.mark.asyncio
async def test_store_video_metadata(tmp_path):
    """Test storing video metadata in database."""
    db_path = tmp_path / "test_catalog.db"
    db = CatalogDatabase(str(db_path))
    
    # Initialize schema
    create_database_schema(str(db_path))
    
    metadata = VideoMetadata(
        video_id="test-video-1",
        original_path="/test/video.mp4",
        file_hash="a1b2c3" + "0" * 58,  # Valid hash
        file_size=5000000,
        duration=300.0
    )
    
    success = await db.store_video_metadata(metadata)
    assert success
    
    # Test duplicate detection
    duplicate = await db.store_video_metadata(metadata)
    assert not duplicate  # Should detect duplicate

# CREATE tests/test_catalog/test_hash_utils.py
def test_calculate_file_hash(tmp_path):
    """Test file hashing with different file sizes."""
    test_file = tmp_path / "test_video.mp4"
    test_content = b"test video content" * 1000  # Create test content
    test_file.write_bytes(test_content)
    
    hash1 = calculate_file_hash(str(test_file))
    hash2 = calculate_file_hash(str(test_file))
    
    assert hash1 == hash2  # Deterministic
    assert len(hash1) == 64  # SHA256 hex length
    assert isinstance(hash1, str)

# CREATE tests/test_catalog/test_video_slicer.py
def test_create_video_slices():
    """Test video slicing with keyword boundaries."""
    transcript = "This discusses price action and market trends. Later we analyze support levels and resistance patterns."
    keywords_data = {
        "price": [{"start": 5.0, "end": 5.5}],
        "support": [{"start": 45.0, "end": 45.8}], 
        "resistance": [{"start": 65.2, "end": 66.0}]
    }
    
    slices = create_video_slices(transcript, keywords_data, 120.0, slice_duration=30.0)
    
    assert len(slices) >= 2  # Should create multiple slices
    assert all(slice.end_time > slice.start_time for slice in slices)
    assert slices[0].start_time == 0.0  # First slice starts at beginning
    assert slices[-1].end_time <= 120.0  # Last slice doesn't exceed video
```

```bash
# Run and iterate until passing:
uv run pytest tests/test_catalog/ -v
# If failing: Read error, understand root cause, fix code, re-run (never mock to pass)
```

### Level 3: Integration Test
```bash
# Test database creation and basic operations
cd /path/to/project
export DATABASE_PATH="test_catalog.db"

# Test schema creation
uv run python -c "
from src.catalog.schema import create_database_schema
create_database_schema('test_catalog.db')
print('Schema created successfully')
"

# Test video ingestion with database storage
uv run python src/main.py ingest --video-path /path/to/test_video.mp4

# Expected: 
# - Database file created with proper schema
# - Video metadata stored in videos table  
# - Video slices created and stored with keywords
# - Search functionality returns relevant results

# Test search functionality
uv run python src/main.py search --keywords "price action" --limit 10

# Expected: JSON output with relevant video slices and timestamps
```

## Final validation Checklist
- [ ] All tests pass: `uv run pytest tests/ -v`
- [ ] No linting errors: `uv run ruff check src/`
- [ ] No type errors: `uv run mypy src/`
- [ ] Database schema creates without errors
- [ ] Video metadata storage and retrieval works correctly
- [ ] Keyword search returns relevant and accurate results
- [ ] File hashing detects duplicates properly
- [ ] Video slicing creates logical segment boundaries
- [ ] Transaction handling prevents data corruption
- [ ] CLI integration replaces markdown output with database storage

---

## Anti-Patterns to Avoid
- ❌ Don't create database connections without proper cleanup (use context managers)
- ❌ Don't store large binary data directly in SQLite (store file paths instead)
- ❌ Don't ignore transaction boundaries - always rollback on errors
- ❌ Don't skip foreign key constraints - they prevent data corruption
- ❌ Don't create indexes after inserting large amounts of data - create them first
- ❌ Don't use string formatting for SQL queries - use parameterized queries
- ❌ Don't load entire video files into memory for hashing - use streaming
- ❌ Don't create video slices without considering keyword boundaries