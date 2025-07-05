name: "Video Clip Clustering and Topic Management"
description: |

## Purpose
This PRP defines the requirements for clustering identified video topics, extracting representative keywords for each cluster, and providing a mechanism for user review and approval of newly discovered topics and their associated clusters. This builds upon the topic identification performed during video ingestion.

## Core Principles
1. **Context is King**: Include ALL necessary documentation, examples, and caveats
2. **Validation Loops**: Provide executable tests/lints the AI can run and fix
3. **Information Dense**: Use keywords and patterns from the codebase
4. **Progressive Success**: Start simple, validate, then enhance
5. **Global rules**: Be sure to follow all rules in GEMINI.md

---

## Goal
To automatically group identified video topics into meaningful clusters, extract representative keywords for each cluster, and provide a mechanism for user review and approval of newly discovered topics and their associated clusters. This will allow users to curate and refine the topic taxonomy, distinguishing between price-action relevant and other topics.

## Why
- Organizes the potentially large number of topics identified during ingestion into manageable, meaningful groups.
- Enables users to curate and refine the topic taxonomy, making the video content more searchable and useful.
- Provides a foundation for filtering and searching video content based on approved topics.
- Facilitates the distinction between price-action relevant topics and general educational content.

## What
The video clip clustering and topic management feature will:
- Retrieve identified topics from video slices stored in the database (output from the video ingestion process).
- Apply agglomerative clustering to group similar topics into clusters.
- Extract representative keywords for each generated cluster.
- Store cluster information (cluster ID, name, keywords, associated topics, approval status) in the local database.
- Implement CLI commands for users to:
    - View newly identified topics and proposed clusters.
    - Approve or reject new topics/clusters.
    - Rename existing clusters.
    - Merge multiple clusters into one.
    - Delete clusters.
    - Manually assign topics to existing clusters.

### Success Criteria
- [ ] Identified topics are successfully grouped into clusters using agglomerative clustering.
- [ ] Meaningful and representative keywords are extracted for each cluster.
- [ ] Cluster information is persistently stored in the database.
- [ ] User can view a list of unapproved topics/clusters via a CLI command.
- [ ] User can approve a new topic/cluster via a CLI command.
- [ ] User can rename, merge, or delete clusters via CLI commands.
- [ ] The clustering process is efficient and scalable for a growing number of topics.

## All Needed Context

### Documentation & References (list all context needed to implement the feature)
```yaml
# MUST READ - Include these in your context window
- url: https://scikit-learn.org/stable/modules/clustering.html#agglomerative-clustering
  why: Official documentation for Agglomerative Clustering in scikit-learn.

- url: https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction
  why: Documentation on text feature extraction (e.g., TF-IDF) for keyword extraction.

- url: https://docs.python.org/3/library/sqlite3.html
  why: Python's built-in SQLite documentation for local database storage.

- file: GEMINI.md
  why: Project-specific guidelines and AI behavior rules.

- file: INITIAL.md
  why: Overall project overview and high-level requirements.

- file: PRPs/video_ingestion_prp.md
  why: Provides context on how video slices and topics are generated and stored.
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
├── PRPs/
│   ├── video_ingestion_prp.md
│   └── templates/
│       └── prp_base.md
└── src/ # (Expected from video_ingestion_prp)
    ├── __init__.py
    ├── main.py
    └── video_ingestion/
        ├── __init__.py
        ├── models.py
        ├── audio_extractor.py
        ├── transcriber.py
        ├── topic_extractor.py
        └── database.py
```

### Desired Codebase tree with files to be added and responsibility of file
```bash
.
├── src/
│   ├── __init__.py
│   ├── main.py
│   ├── video_ingestion/
│   │   ├── models.py       # Updated with TopicCluster model
│   │   └── database.py     # Updated for topic_clusters table
│   ├── topic_management/
│   │   ├── __init__.py
│   │   ├── clusterer.py        # Handles agglomerative clustering
│   │   ├── keyword_extractor.py # Extracts keywords from clusters
│   │   └── topic_manager.py    # Handles user approval, modification of topics/clusters
│   └── cli/
│       └── commands.py     # Updated with new CLI commands for topic management
├── tests/
│   ├── test_topic_management/
│   │   ├── __init__.py
│   │   ├── test_clusterer.py
│   │   ├── test_keyword_extractor.py
│   │   └── test_topic_manager.py
```

### Known Gotchas of our codebase & Library Quirks
```python
# CRITICAL: Choosing appropriate distance metrics and linkage criteria for agglomerative clustering can significantly impact results. Experimentation may be needed.
# CRITICAL: Determining the optimal number of clusters for topic grouping is often subjective and may require user input or heuristic methods.
# CRITICAL: Keyword extraction from clusters needs to be robust to produce meaningful and concise terms. Consider TF-IDF or similar techniques.
# CRITICAL: Designing an intuitive and user-friendly CLI for topic management (approval, merging, renaming, deleting) is essential for adoption.
# CRITICAL: Ensure that changes to topic clusters are reflected correctly in the database and maintain data integrity.
```

## Implementation Blueprint

### Data models and structure

Create the core data models, we ensure type safety and consistency.
```python
# src/video_ingestion/models.py (Update existing)

from pydantic import BaseModel
from typing import List, Dict, Optional

# ... (existing VideoMetadata, VideoSlice, IngestionStatus models) ...

class TopicCluster(BaseModel):
    """Represents a cluster of related topics."""
    cluster_id: str  # Unique ID for the cluster
    name: str        # User-defined or auto-generated name for the cluster
    keywords: List[str] # Representative keywords for the cluster
    topic_ids: List[str] # List of topic IDs belonging to this cluster
    approved: bool   # Whether the cluster has been approved by the user
    created_at: str  # Timestamp of cluster creation
    updated_at: str  # Timestamp of last update
```

### list of tasks to be completed to fullfill the PRP in the order they should be completed

```yaml
Task 1: Update Data Models and Database Schema
MODIFY src/video_ingestion/models.py:
  - Add `TopicCluster` Pydantic model.

MODIFY src/video_ingestion/database.py:
  - Add `topic_clusters` table creation to `init_db`.
  - Add functions to insert, retrieve, update, and delete `TopicCluster` objects.

Task 2: Implement Topic Clustering
CREATE src/topic_management/__init__.py:
CREATE src/topic_management/clusterer.py:
  - Function to perform agglomerative clustering on a list of topics (e.g., using their embeddings or TF-IDF vectors).
  - Input: List of topic strings (or their processed representations).
  - Output: List of cluster assignments or `TopicCluster` objects.

Task 3: Implement Keyword Extraction
CREATE src/topic_management/keyword_extractor.py:
  - Function to extract representative keywords from a group of topics within a cluster.
  - Input: List of topic strings belonging to a cluster.
  - Output: List of keywords.

Task 4: Implement Topic Manager Logic
CREATE src/topic_management/topic_manager.py:
  - Functions to manage topic clusters:
    - `get_unapproved_clusters()`
    - `approve_cluster(cluster_id)`
    - `rename_cluster(cluster_id, new_name)`
    - `merge_clusters(cluster_ids, new_cluster_name)`
    - `delete_cluster(cluster_id)`
    - `assign_topic_to_cluster(topic_id, cluster_id)`

Task 5: Integrate CLI Commands for Topic Management
MODIFY src/cli/commands.py:
  - Add new CLI commands for topic management (e.g., `manage-topics`, `approve-cluster`, `rename-cluster`).
  - Integrate with `topic_management.topic_manager`.

Task 6: Orchestrate Clustering Process
MODIFY src/video_ingestion/topic_extractor.py: (or create a new orchestration module)
  - After initial topic identification, call `clusterer.py` and `keyword_extractor.py`.
  - Store initial `TopicCluster` objects (unapproved) in the database.
```

### Per task pseudocode as needed added to each task
```python

# Task 2: Implement Topic Clustering
# src/topic_management/clusterer.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from typing import List, Tuple

def cluster_topics(topics: List[str], n_clusters: int = None) -> List[int]:
    """
    Clusters a list of topic strings using TF-IDF and Agglomerative Clustering.
    If n_clusters is None, it will attempt to determine an optimal number or use a default.
    """
    if not topics:
        return []

    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(topics)

    # CRITICAL: Consider different linkage and distance metrics
    # Reason: 'ward' is common for Euclidean distance, but others might be better for text.
    # Reason: If n_clusters is not provided, need a strategy to determine it (e.g., silhouette score, elbow method, or a fixed heuristic).
    if n_clusters is None:
        # Placeholder: For initial implementation, let's use a simple heuristic or fixed number
        n_clusters = min(len(topics), 5) # Example: max 5 clusters or number of topics if less

    clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    cluster_labels = clustering.fit_predict(X.toarray()) # .toarray() if using sparse matrix

    return cluster_labels.tolist()

# Task 3: Implement Keyword Extraction
# src/topic_management/keyword_extractor.py
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Dict

def extract_keywords_for_clusters(topics_by_cluster: Dict[int, List[str]], top_n: int = 5) -> Dict[int, List[str]]:
    """
    Extracts top N keywords for each cluster based on TF-IDF.
    """
    cluster_keywords = {}
    for cluster_id, topics_in_cluster in topics_by_cluster.items():
        if not topics_in_cluster:
            cluster_keywords[cluster_id] = []
            continue

        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(topics_in_cluster)
        feature_names = vectorizer.get_feature_names_out()

        # Sum TF-IDF scores for each term across all documents in the cluster
        # Reason: Simple sum can indicate overall importance within the cluster.
        tfidf_scores = tfidf_matrix.sum(axis=0).A1
        sorted_indices = tfidf_scores.argsort()[::-1] # Sort in descending order

        keywords = [feature_names[i] for i in sorted_indices[:top_n]]
        cluster_keywords[cluster_id] = keywords
    return cluster_keywords

# Task 4: Implement Topic Manager Logic (Example: approve_cluster)
# src/topic_management/topic_manager.py
from src.video_ingestion.database import update_topic_cluster_approved_status
from src.video_ingestion.models import TopicCluster
import datetime

def approve_cluster(db_path: str, cluster_id: str) -> bool:
    """
    Approves a topic cluster, marking it as ready for use.
    """
    # CRITICAL: Need a function to retrieve a single cluster from DB first
    # cluster = get_topic_cluster_by_id(db_path, cluster_id)
    # if not cluster:
    #     return False

    # Update the approved status and updated_at timestamp
    # Reason: Ensure data consistency and track changes.
    updated_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    return update_topic_cluster_approved_status(db_path, cluster_id, True, updated_at)

```

### Integration Points
```yaml
DATABASE:
  - migration: "Add table 'topic_clusters' to video_catalog.db"
  - index: "CREATE INDEX idx_topic_cluster_approved ON topic_clusters(approved)"

CLI:
  - add to: src/main.py (CLI entry point)
  - add to: src/cli/commands.py (new commands for topic management)
  - pattern: "parser.add_argument('--approve-cluster', help='Approve a topic cluster by ID.')"
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
# CREATE tests/test_topic_management/test_clusterer.py
import pytest
from src.topic_management.clusterer import cluster_topics

def test_cluster_topics_basic():
    """Test basic clustering functionality."""
    topics = ["market open strategy", "trading psychology", "risk management", "market close strategy", "mindset for trading"]
    labels = cluster_topics(topics, n_clusters=2)
    assert len(labels) == len(topics)
    # Assert that "market open strategy" and "market close strategy" are in the same cluster
    # and "trading psychology" and "mindset for trading" are in another.
    # This requires a more robust check based on expected clustering behavior.
    # For now, a simple check that labels are assigned.
    assert len(set(labels)) <= 2 # Should create at most 2 clusters

def test_cluster_topics_empty():
    """Test clustering with an empty list of topics."""
    labels = cluster_topics([])
    assert labels == []

# CREATE tests/test_topic_management/test_keyword_extractor.py
import pytest
from src.topic_management.keyword_extractor import extract_keywords_for_clusters

def test_extract_keywords_for_clusters_basic():
    """Test keyword extraction for basic clusters."""
    topics_by_cluster = {
        0: ["market open strategy", "market close strategy", "trading session start"],
        1: ["trading psychology", "mindset for trading", "emotional control in trading"]
    }
    keywords = extract_keywords_for_clusters(topics_by_cluster, top_n=2)
    assert 0 in keywords
    assert 1 in keywords
    assert "market" in keywords[0] or "strategy" in keywords[0]
    assert "trading" in keywords[1] or "psychology" in keywords[1]

def test_extract_keywords_for_clusters_empty():
    """Test keyword extraction for empty clusters."""
    topics_by_cluster = {0: [], 1: ["single topic"]}
    keywords = extract_keywords_for_clusters(topics_by_cluster)
    assert keywords[0] == []
    assert len(keywords[1]) > 0

# CREATE tests/test_topic_management/test_topic_manager.py
import pytest
from unittest.mock import patch, MagicMock
from src.topic_management.topic_manager import approve_cluster
from src.video_ingestion.models import TopicCluster
import datetime

@patch('src.video_ingestion.database.update_topic_cluster_approved_status', return_value=True)
def test_approve_cluster_success(mock_update_db):
    """Test successful approval of a cluster."""
    db_path = "dummy.db"
    cluster_id = "cluster123"
    result = approve_cluster(db_path, cluster_id)
    assert result is True
    mock_update_db.assert_called_once_with(db_path, cluster_id, True, MagicMock(spec=str)) # Check for string timestamp

```

```bash
# Run and iterate until passing:
uv run pytest tests/test_topic_management/ -v
# If failing: Read error, understand root cause, fix code, re-run (never mock to pass)
```

### Level 3: Integration Test
```bash
# Manual test:
# 1. Ensure video ingestion has been run and topics are stored in the database.
# 2. Run a CLI command to trigger clustering (will be implemented in Task 6).
#    Example: uv run python src/main.py cluster-topics
# 3. Run a CLI command to view unapproved clusters:
#    Example: uv run python src/main.py manage-topics --list-unapproved
# 4. Run a CLI command to approve a cluster:
#    Example: uv run python src/main.py manage-topics --approve <cluster_id>
# 5. Verify in the database that the cluster's approved status has changed.

# Expected:
# - Clusters are created and stored in the database.
# - CLI commands provide expected output and modify database state correctly.
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
- ❌ Don't hardcode clustering parameters (e.g., number of clusters) without a clear strategy for dynamic determination or user input.
- ❌ Don't rely solely on simple string matching for topic similarity; use vectorization (e.g., TF-IDF, embeddings).
- ❌ Don't neglect edge cases like empty topic lists or clusters with very few topics.
- ❌ Don't make the topic management CLI overly complex; prioritize clear, single-purpose commands.
- ❌ Don't forget to update timestamps (`created_at`, `updated_at`) when modifying cluster data.
