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

    assert len(extracted_keywords) == 3
    assert extracted_keywords[0]["keyword"] == "price"
    assert extracted_keywords[0]["start"] == 1.3
    assert extracted_keywords[0]["end"] == 1.7
    assert extracted_keywords[1]["keyword"] == "action"
    assert extracted_keywords[1]["start"] == 1.7
    assert extracted_keywords[1]["end"] == 2.2
    assert extracted_keywords[2]["keyword"] == "support"
    assert extracted_keywords[2]["start"] == 5.2
    assert extracted_keywords[2]["end"] == 5.8

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

def test_extract_topics_with_timestamps_empty_result():
    """Test with an empty Whisper result."""
    whisper_result = {"text": "", "segments": []}
    extracted_keywords = extract_topics_with_timestamps(whisper_result)
    assert len(extracted_keywords) == 0

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