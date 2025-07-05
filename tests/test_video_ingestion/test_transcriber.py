
import pytest
from unittest.mock import patch, MagicMock
from src.video_ingestion.transcriber import load_whisper_model, transcribe_audio

@patch('src.video_ingestion.transcriber._whisper_model', None)
def test_load_whisper_model_success():
    """Test successful (simulated) loading of Whisper model."""
    with patch('whisper.load_model', return_value=MagicMock()) as mock_load_model:
        model = load_whisper_model()
        mock_load_model.assert_called_once_with("base")
        assert model is not None

@patch('src.video_ingestion.transcriber.load_whisper_model', return_value=MagicMock(transcribe=MagicMock(return_value={"text": "This is a test transcription."})))
def test_transcribe_audio_success(mock_load_model):
    """Test successful (simulated) transcription."""
    transcript = transcribe_audio("dummy_audio.wav")
    assert transcript == "This is a test transcription."
    mock_load_model.assert_called_once()

@patch('src.video_ingestion.transcriber.load_whisper_model', side_effect=Exception("Model load error"))
def test_transcribe_audio_model_load_failure(mock_load_model):
    """Test transcription when model loading fails."""
    with pytest.raises(RuntimeError, match="Whisper transcription failed: Model load error"):
        transcribe_audio("dummy_audio.wav")

@patch('src.video_ingestion.transcriber.load_whisper_model', return_value=MagicMock(transcribe=MagicMock(side_effect=Exception("Transcription error"))))
def test_transcribe_audio_failure(mock_load_model):
    """Test transcription when transcription fails."""
    with pytest.raises(RuntimeError, match="Whisper transcription failed: Transcription error"):
        transcribe_audio("dummy_audio.wav")
