
import spacy
from typing import List, Dict, Any

# Load a spaCy model (e.g., 'en_core_web_sm' or 'en_core_web_md')
# User might need to run: python -m spacy download en_core_web_sm
try:
    _nlp = spacy.load("en_core_web_sm")
except OSError:
    
    _nlp = spacy.load("en_core_web_sm")

def extract_topics_with_timestamps(transcript: str, video_duration_seconds: float) -> List[Dict[str, Any]]:
    """
    Extracts topics and estimates timestamps from a transcript.
    For this spike, we'll do sentence-level segmentation and assign approximate timestamps.
    """
    doc = _nlp(transcript)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    if not sentences:
        return []

    # Estimate time per sentence for basic timestamping
    words_per_second = len(transcript.split()) / video_duration_seconds if video_duration_seconds > 0 else 0
    
    topics_with_timestamps = []
    current_time = 0.0

    for sentence in sentences:
        sentence_words = len(sentence.split())
        estimated_duration = sentence_words / words_per_second if words_per_second > 0 else 0
        
        start_time = current_time
        end_time = current_time + estimated_duration

        # Basic keyword extraction for topic
        keywords = [token.text.lower() for token in _nlp(sentence) if token.pos_ in ["NOUN", "PROPN", "VERB"] and not token.is_stop]
        topic_summary = " ".join(sorted(list(set(keywords)))) # Simple topic summary from keywords

        topics_with_timestamps.append({
            "topic": topic_summary if topic_summary else sentence[:50] + "...", # Fallback if no keywords
            "start_time": round(start_time, 2),
            "end_time": round(end_time, 2),
            "text_segment": sentence
        })
        current_time = end_time
    
    return topics_with_timestamps
