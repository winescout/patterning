import pytest
from src.video_ingestion.topic_extractor import extract_topics_with_timestamps

def test_extract_topics_with_timestamps_basic():
    """Test basic keyword extraction and timestamping from a Whisper-like result."""
    whisper_result = {
        "text": "This is a test about price action and support levels.",
        "segments": [
            {
                "id": 0,
                "start": 0.0,
                "end": 5.0,
                "text": "This is a test about price action",
                "words": [
                    {"word": "This", "start": 0.0, "end": 0.5},
                    {"word": "is", "start": 0.5, "end": 0.6},
                    {"word": "a", "start": 0.6, "end": 0.7},
                    {"word": "test", "start": 0.7, "end": 1.0},
                    {"word": "about", "start": 1.0, "end": 1.3},
                    {"word": "price", "start": 1.3, "end": 1.7},
                    {"word": "action", "start": 1.7, "end": 2.2}
                ]
            },
            {
                "id": 1,
                "start": 5.0,
                "end": 8.0,
                "text": "and support levels.",
                "words": [
                    {"word": "and", "start": 5.0, "end": 5.2},
                    {"word": "support", "start": 5.2, "end": 5.8},
                    {"word": "levels.", "start": 5.8, "end": 6.5}
                ]
            }
        ]
    }

    extracted_keywords = extract_topics_with_timestamps(whisper_result)

    # Check that we have the expected keywords
    # Should find: price, action, support, and level (lemma of "levels.")
    assert len(extracted_keywords) == 4
    assert "price" in extracted_keywords
    assert "action" in extracted_keywords
    assert "support" in extracted_keywords
    assert "level" in extracted_keywords
    
    # Check timestamp format for each keyword
    assert len(extracted_keywords["price"]) == 1
    assert extracted_keywords["price"][0]["start"] == 1.3
    assert extracted_keywords["price"][0]["end"] == 1.7
    
    assert len(extracted_keywords["action"]) == 1
    assert extracted_keywords["action"][0]["start"] == 1.7
    assert extracted_keywords["action"][0]["end"] == 2.2
    
    assert len(extracted_keywords["support"]) == 1
    assert extracted_keywords["support"][0]["start"] == 5.2
    assert extracted_keywords["support"][0]["end"] == 5.8
    
    assert len(extracted_keywords["level"]) == 1
    assert extracted_keywords["level"][0]["start"] == 5.8
    assert extracted_keywords["level"][0]["end"] == 6.5

def test_extract_topics_with_timestamps_no_keywords():
    """Test with a transcript containing no keywords of interest."""
    whisper_result = {
        "text": "This is a general conversation.",
        "segments": [
            {
                "id": 0,
                "start": 0.0,
                "end": 3.0,
                "text": "This is a general conversation.",
                "words": [
                    {"word": "This", "start": 0.0, "end": 0.5},
                    {"word": "is", "start": 0.5, "end": 0.6},
                    {"word": "a", "start": 0.6, "end": 0.7},
                    {"word": "general", "start": 0.7, "end": 1.2},
                    {"word": "conversation.", "start": 1.2, "end": 2.5}
                ]
            }
        ]
    }
    extracted_keywords = extract_topics_with_timestamps(whisper_result)
    assert len(extracted_keywords) == 0
    assert isinstance(extracted_keywords, dict)

def test_extract_topics_with_timestamps_empty_result():
    """Test with an empty Whisper result."""
    whisper_result = {"text": "", "segments": []}
    extracted_keywords = extract_topics_with_timestamps(whisper_result)
    assert len(extracted_keywords) == 0
    assert isinstance(extracted_keywords, dict)

def test_extract_topics_with_timestamps_no_words_in_segment():
    """Test with segments that have no word information."""
    whisper_result = {
        "text": "Some text.",
        "segments": [
            {
                "id": 0,
                "start": 0.0,
                "end": 1.0,
                "text": "Some text.",
                "words": []
            }
        ]
    }
    extracted_keywords = extract_topics_with_timestamps(whisper_result)
    assert len(extracted_keywords) == 0
    assert isinstance(extracted_keywords, dict)

def test_extract_topics_with_timestamps_multiple_occurrences():
    """Test keywords appearing multiple times with different timestamps."""
    whisper_result = {
        "text": "Price goes up, then price goes down.",
        "segments": [
            {
                "id": 0,
                "start": 0.0,
                "end": 6.0,
                "text": "Price goes up, then price goes down.",
                "words": [
                    {"word": "Price", "start": 0.0, "end": 0.5},
                    {"word": "goes", "start": 0.5, "end": 0.8},
                    {"word": "up,", "start": 0.8, "end": 1.0},
                    {"word": "then", "start": 1.0, "end": 1.3},
                    {"word": "price", "start": 1.3, "end": 1.7},
                    {"word": "goes", "start": 1.7, "end": 2.0},
                    {"word": "down.", "start": 2.0, "end": 2.5}
                ]
            }
        ]
    }
    extracted_keywords = extract_topics_with_timestamps(whisper_result)
    
    # Should find "price" twice (case-insensitive)
    assert "price" in extracted_keywords
    assert len(extracted_keywords["price"]) == 2
    assert extracted_keywords["price"][0]["start"] == 0.0
    assert extracted_keywords["price"][0]["end"] == 0.5
    assert extracted_keywords["price"][1]["start"] == 1.3
    assert extracted_keywords["price"][1]["end"] == 1.7