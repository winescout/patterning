
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
