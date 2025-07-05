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