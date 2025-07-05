
import pytest
from unittest.mock import patch
from src.video_ingestion.transcriber import load_video_llama_model, transcribe_audio

@patch('src.video_ingestion.transcriber._video_llama_model', None)
@patch('src.video_ingestion.transcriber._video_llama_processor', None)
def test_load_video_llama_model_success():
    """Test successful (simulated) loading of Video-Llama model."""
    model, processor = load_video_llama_model()
    assert model is True
    assert processor is True

@patch('src.video_ingestion.transcriber.load_video_llama_model', return_value=(True, True))
def test_transcribe_audio_success(mock_load_model):
    """Test successful (simulated) transcription."""
    transcript = transcribe_audio("dummy_audio.wav")
    assert "transcription" in transcript and "Video-Llama" in transcript
    mock_load_model.assert_called_once()

@patch('src.video_ingestion.transcriber.load_video_llama_model', side_effect=RuntimeError("Failed to load Video-Llama model. Ensure it's installed and configured correctly: Model init error"))
def test_transcribe_audio_model_load_failure(mock_load_model):
    """Test transcription when model loading fails."""
    with pytest.raises(RuntimeError, match="Failed to load Video-Llama model. Ensure it's installed and configured correctly: Model init error"):
        transcribe_audio("dummy_audio.wav")
