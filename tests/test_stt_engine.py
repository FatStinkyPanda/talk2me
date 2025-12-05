import os
import sys
from unittest.mock import Mock, mock_open, patch

import numpy as np
import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from talk2me.stt.engine import STTEngine


class TestSTTEngine:
    """Test suite for STTEngine class."""

    @pytest.fixture
    def mock_config(self):
        """Mock configuration data."""
        return {"stt": {"model_path": "/path/to/model", "sample_rate": 16000}}

    @pytest.fixture
    def mock_vosk_model(self):
        """Mock Vosk Model."""
        return Mock()

    @pytest.fixture
    def mock_vosk_recognizer(self):
        """Mock Vosk KaldiRecognizer."""
        return Mock()

    def test_init_success(self, mock_config, mock_vosk_model, mock_vosk_recognizer):
        """Test successful initialization."""
        with (
            patch("builtins.open", mock_open(read_data="")),
            patch("yaml.safe_load", return_value=mock_config),
            patch("talk2me.stt.engine.Model", return_value=mock_vosk_model),
            patch("talk2me.stt.engine.KaldiRecognizer", return_value=mock_vosk_recognizer),
        ):
            engine = STTEngine()

            assert engine.model == mock_vosk_model
            assert engine.sample_rate == 16000

    def test_init_config_file_not_found(self):
        """Test initialization when config file is not found."""
        with (
            patch("builtins.open", side_effect=FileNotFoundError),
            pytest.raises(FileNotFoundError),
        ):
            STTEngine("nonexistent.yaml")

    def test_init_yaml_error(self):
        """Test initialization when YAML parsing fails."""
        with (
            patch("builtins.open", mock_open(read_data="invalid: yaml: [")),
            patch("yaml.safe_load", side_effect=yaml.YAMLError("YAML error")),
            pytest.raises(RuntimeError, match="Invalid configuration file"),
        ):
            STTEngine()

    def test_init_missing_model_path(self):
        """Test initialization with missing model path."""
        config = {"stt": {}}
        with (
            patch("builtins.open", mock_open(read_data="")),
            patch("yaml.safe_load", return_value=config),
            pytest.raises(ValueError, match="STT model_path not specified"),
        ):
            STTEngine()

    def test_init_model_load_failure(self, mock_config):
        """Test initialization when model loading fails."""
        with (
            patch("builtins.open", mock_open(read_data="")),
            patch("yaml.safe_load", return_value=mock_config),
            patch("talk2me.stt.engine.Model", side_effect=Exception("Model load failed")),
            pytest.raises(RuntimeError, match="Model loading failed"),
        ):
            STTEngine()

    def test_init_config_empty(self):
        """Test initialization when config file is empty (yaml.safe_load returns None)."""
        with (
            patch("builtins.open", mock_open(read_data="")),
            patch("yaml.safe_load", return_value=None),
            pytest.raises(ValueError, match="STT model_path not specified"),
        ):
            STTEngine()

    def test_transcribe_bytes_success(self, mock_config, mock_vosk_model):
        """Test successful transcription with bytes input."""
        # Mock recognizer behavior
        mock_recognizer = Mock()
        mock_recognizer.AcceptWaveform.side_effect = [True, False, True]  # Two chunks, final
        mock_recognizer.Result.return_value = '{"text": "hello"}'
        mock_recognizer.FinalResult.return_value = '{"text": "world"}'

        with (
            patch("builtins.open", mock_open(read_data="")),
            patch("yaml.safe_load", return_value=mock_config),
            patch("talk2me.stt.engine.Model", return_value=mock_vosk_model),
            patch("talk2me.stt.engine.KaldiRecognizer", return_value=mock_recognizer),
        ):
            engine = STTEngine()

            result = engine.transcribe(b"audio_data")

            assert result == "hello world"
            assert mock_recognizer.AcceptWaveform.call_count == 3  # Two chunks + final check

    def test_transcribe_numpy_array_success(self, mock_config, mock_vosk_model):
        """Test successful transcription with numpy array input."""
        mock_recognizer = Mock()
        mock_recognizer.AcceptWaveform.return_value = True
        mock_recognizer.Result.return_value = '{"text": "test"}'
        mock_recognizer.FinalResult.return_value = '{"text": ""}'

        with (
            patch("builtins.open", mock_open(read_data="")),
            patch("yaml.safe_load", return_value=mock_config),
            patch("talk2me.stt.engine.Model", return_value=mock_vosk_model),
            patch("talk2me.stt.engine.KaldiRecognizer", return_value=mock_recognizer),
        ):
            engine = STTEngine()

            audio_array = np.array([0.1, 0.2, 0.3], dtype=np.float32)
            result = engine.transcribe(audio_array)

            assert result == "test"
            # Should convert float32 to int16
            mock_recognizer.AcceptWaveform.assert_called()

    def test_transcribe_with_custom_sample_rate(self, mock_config, mock_vosk_model):
        """Test transcription with custom sample rate."""
        mock_recognizer = Mock()
        mock_recognizer.AcceptWaveform.return_value = True
        mock_recognizer.Result.return_value = '{"text": "test"}'
        mock_recognizer.FinalResult.return_value = '{"text": ""}'

        with (
            patch("builtins.open", mock_open(read_data="")),
            patch("yaml.safe_load", return_value=mock_config),
            patch("talk2me.stt.engine.Model", return_value=mock_vosk_model),
            patch("talk2me.stt.engine.KaldiRecognizer") as mock_kaldi,
        ):
            mock_kaldi.return_value = mock_recognizer
            engine = STTEngine()

            result = engine.transcribe(b"audio_data", sample_rate=22050)

            assert result == "test"
            # Should use custom sample rate
            mock_kaldi.assert_called_with(mock_vosk_model, 22050)

    def test_transcribe_invalid_audio_type(
        self, mock_config, mock_vosk_model, mock_vosk_recognizer
    ):
        """Test transcription with invalid audio type."""
        with (
            patch("builtins.open", mock_open(read_data="")),
            patch("yaml.safe_load", return_value=mock_config),
            patch("talk2me.stt.engine.Model", return_value=mock_vosk_model),
            patch("talk2me.stt.engine.KaldiRecognizer", return_value=mock_vosk_recognizer),
        ):
            engine = STTEngine()

            with pytest.raises(ValueError, match="Audio data must be bytes or numpy array"):
                engine.transcribe("invalid_audio")

    def test_transcribe_vosk_error(self, mock_config, mock_vosk_model):
        """Test transcription when Vosk processing fails."""
        mock_recognizer = Mock()
        mock_recognizer.AcceptWaveform.side_effect = Exception("Vosk error")

        with (
            patch("builtins.open", mock_open(read_data="")),
            patch("yaml.safe_load", return_value=mock_config),
            patch("talk2me.stt.engine.Model", return_value=mock_vosk_model),
            patch("talk2me.stt.engine.KaldiRecognizer", return_value=mock_recognizer),
        ):
            engine = STTEngine()

            with pytest.raises(RuntimeError, match="Transcription failed"):
                engine.transcribe(b"audio_data")

    def test_transcribe_empty_result(self, mock_config, mock_vosk_model):
        """Test transcription with empty results."""
        mock_recognizer = Mock()
        mock_recognizer.AcceptWaveform.return_value = True
        mock_recognizer.Result.return_value = '{"text": ""}'
        mock_recognizer.FinalResult.return_value = '{"text": ""}'

        with (
            patch("builtins.open", mock_open(read_data="")),
            patch("yaml.safe_load", return_value=mock_config),
            patch("talk2me.stt.engine.Model", return_value=mock_vosk_model),
            patch("talk2me.stt.engine.KaldiRecognizer", return_value=mock_recognizer),
        ):
            engine = STTEngine()

            result = engine.transcribe(b"audio_data")

            assert result == ""

    def test_transcribe_partial_results(self, mock_config, mock_vosk_model):
        """Test transcription with partial results."""
        mock_recognizer = Mock()
        # First call returns result, second doesn't, third for final
        mock_recognizer.AcceptWaveform.side_effect = [True, False, True]
        mock_recognizer.Result.return_value = '{"text": "hello"}'
        mock_recognizer.FinalResult.return_value = '{"text": "world"}'

        with (
            patch("builtins.open", mock_open(read_data="")),
            patch("yaml.safe_load", return_value=mock_config),
            patch("talk2me.stt.engine.Model", return_value=mock_vosk_model),
            patch("talk2me.stt.engine.KaldiRecognizer", return_value=mock_recognizer),
        ):
            engine = STTEngine()

            result = engine.transcribe(b"audio_data" * 1000)  # Make it long enough for chunks

            assert result == "hello world"

    def test_transcribe_multiple_chunks(self, mock_config, mock_vosk_model):
        """Test transcription with multiple audio chunks."""
        mock_recognizer = Mock()
        mock_recognizer.AcceptWaveform.side_effect = [True, True, True]  # Three chunks
        mock_recognizer.Result.side_effect = [
            '{"text": "hello"}',
            '{"text": "beautiful"}',
            '{"text": "world"}',
        ]
        mock_recognizer.FinalResult.return_value = '{"text": ""}'

        with (
            patch("builtins.open", mock_open(read_data="")),
            patch("yaml.safe_load", return_value=mock_config),
            patch("talk2me.stt.engine.Model", return_value=mock_vosk_model),
            patch("talk2me.stt.engine.KaldiRecognizer", return_value=mock_recognizer),
        ):
            engine = STTEngine()

            # Create large audio data to trigger chunking
            large_audio = b"audio_data" * 3000
            result = engine.transcribe(large_audio)

            assert result == "hello beautiful world"

    def test_transcribe_numpy_conversion_warning(self, mock_config, mock_vosk_model):
        """Test numpy array conversion with warning."""
        mock_recognizer = Mock()
        mock_recognizer.AcceptWaveform.return_value = True
        mock_recognizer.Result.return_value = '{"text": "test"}'
        mock_recognizer.FinalResult.return_value = '{"text": ""}'

        with (
            patch("builtins.open", mock_open(read_data="")),
            patch("yaml.safe_load", return_value=mock_config),
            patch("talk2me.stt.engine.Model", return_value=mock_vosk_model),
            patch("talk2me.stt.engine.KaldiRecognizer", return_value=mock_recognizer),
            patch("talk2me.stt.engine.logger") as mock_logger,
        ):
            engine = STTEngine()

            # Float64 array should trigger conversion warning
            audio_array = np.array([0.1, 0.2, 0.3], dtype=np.float64)
            result = engine.transcribe(audio_array)

            assert result == "test"
            mock_logger.warning.assert_called_with("Converting audio data to int16")

    def test_transcribe_empty_audio(self, mock_config, mock_vosk_model, mock_vosk_recognizer):
        """Test transcription with empty audio data."""
        mock_vosk_recognizer.FinalResult.return_value = '{"text": ""}'

        with (
            patch("builtins.open", mock_open(read_data="")),
            patch("yaml.safe_load", return_value=mock_config),
            patch("talk2me.stt.engine.Model", return_value=mock_vosk_model),
            patch("talk2me.stt.engine.KaldiRecognizer", return_value=mock_vosk_recognizer),
        ):
            engine = STTEngine()

            result = engine.transcribe(b"")

            assert result == ""

    def test_transcribe_invalid_json_result(self, mock_config, mock_vosk_model):
        """Test transcription when Vosk returns invalid JSON."""
        mock_recognizer = Mock()
        mock_recognizer.AcceptWaveform.return_value = True
        mock_recognizer.Result.return_value = "invalid json"
        mock_recognizer.FinalResult.return_value = '{"text": ""}'

        with (
            patch("builtins.open", mock_open(read_data="")),
            patch("yaml.safe_load", return_value=mock_config),
            patch("talk2me.stt.engine.Model", return_value=mock_vosk_model),
            patch("talk2me.stt.engine.KaldiRecognizer", return_value=mock_recognizer),
        ):
            engine = STTEngine()

            with pytest.raises(RuntimeError, match="Transcription failed"):
                engine.transcribe(b"audio_data")
