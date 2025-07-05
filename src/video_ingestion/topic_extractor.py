import spacy
from typing import List, Dict, Any
from collections import defaultdict

# Load a spaCy model (e.g., 'en_core_web_sm' or 'en_core_web_md')
# User might need to run: python -m spacy download en_core_web_sm
try:
    _nlp = spacy.load("en_core_web_sm")
except OSError:
    raise RuntimeError("spaCy model 'en_core_web_sm' not found. Please install it with: python -m spacy download en_core_web_sm")

def extract_topics_with_timestamps(whisper_result: Dict[str, Any]) -> Dict[str, List[Dict[str, float]]]:
    """
    Extracts specific keywords and their timestamps from a Whisper transcription result.
    Uses spaCy for enhanced NLP processing to identify trading/financial terms.
    
    Args:
        whisper_result: Whisper transcription result with segments and word-level timestamps
        
    Returns:
        Dictionary mapping keywords to list of their timestamp occurrences
        Format: {"keyword": [{"start": 1.2, "end": 1.5}, {"start": 10.3, "end": 10.7}]}
    """
    # Enhanced keyword list with trading/financial terms
    keywords_of_interest = {
        "price", "action", "support", "resistance", "trend", "breakout",
        "candlestick", "chart", "indicator", "moving", "average", "RSI", "MACD",
        "volume", "pattern", "bearish", "bullish", "long", "short", "trade",
        "buy", "sell", "market", "stock", "profit", "loss", "risk", "analysis",
        "technical", "fundamental", "momentum", "oversold", "overbought", "signal",
        "entry", "exit", "stop", "target", "fibonacci", "bollinger", "bands",
        "divergence", "convergence", "channel", "trendline", "level", "zone"
    }
    
    # Dictionary to store keywords and their timestamps
    keyword_timestamps = defaultdict(list)
    
    if "segments" in whisper_result:
        for segment in whisper_result["segments"]:
            if "words" in segment:
                # Process segment text with spaCy for enhanced analysis
                segment_text = segment.get("text", "")
                if segment_text:
                    doc = _nlp(segment_text.lower())
                    
                    # Extract lemmatized tokens (base forms of words)
                    lemmatized_tokens = {token.lemma_ for token in doc if not token.is_stop and token.is_alpha}
                    
                    # Find matches with our keywords of interest
                    for word_info in segment["words"]:
                        word = word_info["word"].lower().strip()
                        # Remove punctuation for cleaner matching
                        clean_word = word.rstrip('.,!?;:')
                        
                        matched_keyword = None
                        
                        # Direct keyword match
                        if clean_word in keywords_of_interest:
                            matched_keyword = clean_word
                        else:
                            # Check lemmatized form for better matching (e.g., "trading" -> "trade")
                            word_doc = _nlp(clean_word)
                            if word_doc and word_doc[0].lemma_ in keywords_of_interest:
                                matched_keyword = word_doc[0].lemma_
                        
                        # Add the match (avoid duplicates)
                        if matched_keyword:
                            keyword_timestamps[matched_keyword].append({
                                "start": word_info["start"],
                                "end": word_info["end"]
                            })
    
    # Convert defaultdict to regular dict for cleaner output
    return dict(keyword_timestamps)