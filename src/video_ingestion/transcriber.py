
import os
try:
    import torch
except ImportError:
    torch = None # type: ignore
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
        if torch is None:
            raise RuntimeError("PyTorch is not installed. Please install it to use Whisper.")
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
    try:
        model = load_whisper_model()
        print(f"Transcribing audio from {audio_path} using Whisper...")
        result = model.transcribe(audio_path)
        return result["text"]
    except Exception as e:
        raise RuntimeError(f"Whisper transcription failed: {e}")
