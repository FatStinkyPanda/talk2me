import os
import sys
from unittest.mock import Mock, mock_open, patch

import numpy as np
import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from talk2me.core.wake_word import WakeWordDetector


class TestWakeWordDetector:
    """Test suite for WakeWordDetector class."""

    @pytest.fixture
    def mock_config(self):
        """Mock configuration data."""
        return {
            "stt": {
                "wake_word_model_path": "models/vosk-model-small-en-us-0.15",
                "sample_rate": 16000,
            },
            "wake_words": {"activation": ["hey", "hello", "wake up"]},
        }

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
            patch("talk2me.core.wake_word.Model", return_value=mock_vosk_model),
            patch("talk2me.core.wake_word.KaldiRecognizer", return_value=mock_vosk_recognizer),
        ):
            detector = WakeWordDetector()

            assert detector.model == mock_vosk_model
            assert detector.sample_rate == 16000
            assert detector.wake_words == ["hey", "hello", "wake up"]
            assert detector.recognizer == mock_vosk_recognizer
            assert not detector.detected
            assert not detector.listening
            assert detector.thread is None

    def test_init_config_file_not_found(self):
        """Test initialization when config file is not found."""
        with (
            patch("builtins.open", side_effect=FileNotFoundError),
            pytest.raises(FileNotFoundError),
        ):
            WakeWordDetector("nonexistent.yaml")

    def test_init_yaml_error(self):
        """Test initialization when YAML parsing fails."""
        with (
            patch("builtins.open", mock_open(read_data="invalid: yaml: [")),
            patch("yaml.safe_load", side_effect=yaml.YAMLError("YAML error")),
            pytest.raises(RuntimeError, match="Invalid configuration file"),
        ):
            WakeWordDetector()

    def test_init_missing_model_path(self):
        """Test initialization with missing model path."""
        config = {"stt": {}, "wake_words": {"activation": ["hey"]}}
        with (
            patch("builtins.open", mock_open(read_data="")),
            patch("yaml.safe_load", return_value=config),
            pytest.raises(ValueError, match="Wake word model_path not specified"),
        ):
            WakeWordDetector()

    def test_init_missing_wake_words(self):
        """Test initialization with missing wake words."""
        config = {"stt": {"wake_word_model_path": "/path"}, "wake_words": {}}
        with (
            patch("builtins.open", mock_open(read_data="")),
            patch("yaml.safe_load", return_value=config),
            pytest.raises(ValueError, match="Wake words not specified"),
        ):
            WakeWordDetector()

    def test_init_model_load_failure(self, mock_config):
        """Test initialization when model loading fails."""
        with (
            patch("builtins.open", mock_open(read_data="")),
            patch("yaml.safe_load", return_value=mock_config),
            patch("talk2me.core.wake_word.Model", side_effect=Exception("Model load failed")),
            pytest.raises(RuntimeError, match="Model loading failed"),
        ):
            WakeWordDetector()

    def test_start_listening_success(self, mock_config, mock_vosk_model, mock_vosk_recognizer):
        """Test successful start listening."""
        with (
            patch("builtins.open", mock_open(read_data="")),
            patch("yaml.safe_load", return_value=mock_config),
            patch("talk2me.core.wake_word.Model", return_value=mock_vosk_model),
            patch("talk2me.core.wake_word.KaldiRecognizer", return_value=mock_vosk_recognizer),
            patch("threading.Thread") as mock_thread,
        ):
            detector = WakeWordDetector()
            detector.start_listening()

            assert detector.listening
            assert not detector.detected
            mock_thread.assert_called_once()
            mock_thread.return_value.start.assert_called_once()

    def test_start_listening_already_running(
        self, mock_config, mock_vosk_model, mock_vosk_recognizer
    ):
        """Test start listening when already running."""
        with (
            patch("builtins.open", mock_open(read_data="")),
            patch("yaml.safe_load", return_value=mock_config),
            patch("talk2me.core.wake_word.Model", return_value=mock_vosk_model),
            patch("talk2me.core.wake_word.KaldiRecognizer", return_value=mock_vosk_recognizer),
        ):
            detector = WakeWordDetector()
            detector.listening = True

            with patch("talk2me.core.wake_word.logger") as mock_logger:
                detector.start_listening()
                mock_logger.warning.assert_called_once_with(
                    "Wake word detection is already running"
                )

    def test_process_audio_wake_word_in_final(self, mock_config, mock_vosk_model):
        """Test audio processing with wake word in final result."""
        mock_recognizer = Mock()
        mock_recognizer.AcceptWaveform.return_value = True
        mock_recognizer.Result.return_value = '{"text": "hey there"}'

        with (
            patch("builtins.open", mock_open(read_data="")),
            patch("yaml.safe_load", return_value=mock_config),
            patch("talk2me.core.wake_word.Model", return_value=mock_vosk_model),
            patch("talk2me.core.wake_word.KaldiRecognizer", return_value=mock_recognizer),
            patch("talk2me.core.wake_word.logger") as mock_logger,
        ):
            detector = WakeWordDetector()
            detector.recognizer = mock_recognizer
            detector.listening = True
            detector.audio_queue.put(b"audio")

            # Simulate processing
            detector._process_audio()

            # Should detect wake word
            assert detector.detected
            mock_logger.info.assert_called_with("Wake word detected: 'hey there'")

    def test_process_audio_wake_word_in_partial(self, mock_config, mock_vosk_model):
        """Test audio processing with wake word in partial result."""
        mock_recognizer = Mock()
        mock_recognizer.AcceptWaveform.return_value = False
        mock_recognizer.PartialResult.return_value = '{"partial": "hey listen"}'

        with (
            patch("builtins.open", mock_open(read_data="")),
            patch("yaml.safe_load", return_value=mock_config),
            patch("talk2me.core.wake_word.Model", return_value=mock_vosk_model),
            patch("talk2me.core.wake_word.KaldiRecognizer", return_value=mock_recognizer),
            patch("talk2me.core.wake_word.logger") as mock_logger,
        ):
            detector = WakeWordDetector()
            detector.recognizer = mock_recognizer
            detector.listening = True
            detector.audio_queue.put(b"audio")

            # Simulate processing
            detector._process_audio()

            # Should detect wake word
            assert detector.detected
            mock_logger.info.assert_called_with("Wake word detected in partial: 'hey listen'")

    def test_process_audio_no_wake_word(self, mock_config, mock_vosk_model):
        """Test audio processing with no wake word."""
        mock_recognizer = Mock()
        mock_recognizer.AcceptWaveform.return_value = True
        mock_recognizer.Result.return_value = '{"text": "some text"}'

        with (
            patch("builtins.open", mock_open(read_data="")),
            patch("yaml.safe_load", return_value=mock_config),
            patch("talk2me.core.wake_word.Model", return_value=mock_vosk_model),
            patch("talk2me.core.wake_word.KaldiRecognizer", return_value=mock_recognizer),
        ):
            detector = WakeWordDetector()
            detector.recognizer = mock_recognizer
            detector.listening = True
            detector.audio_queue.put(b"audio")

            # Simulate processing
            detector._process_audio()

            # Should not detect wake word
            assert not detector.detected

    def test_process_audio_error(self, mock_config, mock_vosk_model):
        """Test audio processing with error."""
        mock_recognizer = Mock()
        mock_recognizer.AcceptWaveform.side_effect = Exception("Processing error")

        with (
            patch("builtins.open", mock_open(read_data="")),
            patch("yaml.safe_load", return_value=mock_config),
            patch("talk2me.core.wake_word.Model", return_value=mock_vosk_model),
            patch("talk2me.core.wake_word.KaldiRecognizer", return_value=mock_recognizer),
            patch("talk2me.core.wake_word.logger") as mock_logger,
        ):
            detector = WakeWordDetector()
            detector.recognizer = mock_recognizer
            detector.listening = True
            detector.audio_queue.put(b"audio")

            # Simulate processing
            detector._process_audio()

            mock_logger.error.assert_called_with("Error processing audio chunk: Processing error")

    def test_check_wake_word_found(self, mock_config, mock_vosk_model, mock_vosk_recognizer):
        """Test wake word checking when found."""
        with (
            patch("builtins.open", mock_open(read_data="")),
            patch("yaml.safe_load", return_value=mock_config),
            patch("talk2me.core.wake_word.Model", return_value=mock_vosk_model),
            patch("talk2me.core.wake_word.KaldiRecognizer", return_value=mock_vosk_recognizer),
        ):
            detector = WakeWordDetector()

            assert detector._check_wake_word("hey there") is True
            assert detector._check_wake_word("hello world") is True
            assert detector._check_wake_word("wake up now") is True

    def test_check_wake_word_not_found(self, mock_config, mock_vosk_model, mock_vosk_recognizer):
        """Test wake word checking when not found."""
        with (
            patch("builtins.open", mock_open(read_data="")),
            patch("yaml.safe_load", return_value=mock_config),
            patch("talk2me.core.wake_word.Model", return_value=mock_vosk_model),
            patch("talk2me.core.wake_word.KaldiRecognizer", return_value=mock_vosk_recognizer),
        ):
            detector = WakeWordDetector()

            assert detector._check_wake_word("some text") is False
            assert detector._check_wake_word("") is False

    def test_convert_audio_to_bytes_from_bytes(
        self, mock_config, mock_vosk_model, mock_vosk_recognizer
    ):
        """Test audio conversion from bytes."""
        with (
            patch("builtins.open", mock_open(read_data="")),
            patch("yaml.safe_load", return_value=mock_config),
            patch("talk2me.core.wake_word.Model", return_value=mock_vosk_model),
            patch("talk2me.core.wake_word.KaldiRecognizer", return_value=mock_vosk_recognizer),
        ):
            detector = WakeWordDetector()

            result = detector._convert_audio_to_bytes(b"audio_data")
            assert result == b"audio_data"

    def test_convert_audio_to_bytes_from_numpy_int16(
        self, mock_config, mock_vosk_model, mock_vosk_recognizer
    ):
        """Test audio conversion from numpy int16 array."""
        with (
            patch("builtins.open", mock_open(read_data="")),
            patch("yaml.safe_load", return_value=mock_config),
            patch("talk2me.core.wake_word.Model", return_value=mock_vosk_model),
            patch("talk2me.core.wake_word.KaldiRecognizer", return_value=mock_vosk_recognizer),
        ):
            detector = WakeWordDetector()

            audio_array = np.array([1, 2, 3], dtype=np.int16)
            result = detector._convert_audio_to_bytes(audio_array)
            assert result == b"\x01\x00\x02\x00\x03\x00"

    def test_convert_audio_to_bytes_from_numpy_float(
        self, mock_config, mock_vosk_model, mock_vosk_recognizer
    ):
        """Test audio conversion from numpy float array."""
        with (
            patch("builtins.open", mock_open(read_data="")),
            patch("yaml.safe_load", return_value=mock_config),
            patch("talk2me.core.wake_word.Model", return_value=mock_vosk_model),
            patch("talk2me.core.wake_word.KaldiRecognizer", return_value=mock_vosk_recognizer),
            patch("talk2me.core.wake_word.logger") as mock_logger,
        ):
            detector = WakeWordDetector()

            audio_array = np.array([0.1, 0.2, 0.3], dtype=np.float32)
            result = detector._convert_audio_to_bytes(audio_array)

            # Should convert to int16
            expected = (audio_array * 32767).astype(np.int16).tobytes()
            assert result == expected
            mock_logger.warning.assert_called_once()

    def test_convert_audio_to_bytes_invalid_type(
        self, mock_config, mock_vosk_model, mock_vosk_recognizer
    ):
        """Test audio conversion with invalid type."""
        with (
            patch("builtins.open", mock_open(read_data="")),
            patch("yaml.safe_load", return_value=mock_config),
            patch("talk2me.core.wake_word.Model", return_value=mock_vosk_model),
            patch("talk2me.core.wake_word.KaldiRecognizer", return_value=mock_vosk_recognizer),
        ):
            detector = WakeWordDetector()

            with pytest.raises(ValueError, match="Invalid audio chunk type"):
                detector._convert_audio_to_bytes("invalid")

    def test_process_vosk_result(self, mock_config, mock_vosk_model, mock_vosk_recognizer):
        """Test Vosk result processing."""
        with (
            patch("builtins.open", mock_open(read_data="")),
            patch("yaml.safe_load", return_value=mock_config),
            patch("talk2me.core.wake_word.Model", return_value=mock_vosk_model),
            patch("talk2me.core.wake_word.KaldiRecognizer", return_value=mock_vosk_recognizer),
        ):
            detector = WakeWordDetector()

            result = detector._process_vosk_result('{"text": "hello world"}')
            assert result == "hello world"

    def test_process_vosk_partial(self, mock_config, mock_vosk_model, mock_vosk_recognizer):
        """Test Vosk partial result processing."""
        with (
            patch("builtins.open", mock_open(read_data="")),
            patch("yaml.safe_load", return_value=mock_config),
            patch("talk2me.core.wake_word.Model", return_value=mock_vosk_model),
            patch("talk2me.core.wake_word.KaldiRecognizer", return_value=mock_vosk_recognizer),
        ):
            detector = WakeWordDetector()

            result = detector._process_vosk_partial('{"partial": "hello"}')
            assert result == "hello"

    def test_add_audio_chunk_when_listening(
        self, mock_config, mock_vosk_model, mock_vosk_recognizer
    ):
        """Test adding audio chunk when listening."""
        with (
            patch("builtins.open", mock_open(read_data="")),
            patch("yaml.safe_load", return_value=mock_config),
            patch("talk2me.core.wake_word.Model", return_value=mock_vosk_model),
            patch("talk2me.core.wake_word.KaldiRecognizer", return_value=mock_vosk_recognizer),
        ):
            detector = WakeWordDetector()
            detector.listening = True

            detector.add_audio_chunk(b"audio_data")

            # Check that audio was added to queue
            assert detector.audio_queue.get() == b"audio_data"

    def test_add_audio_chunk_when_not_listening(
        self, mock_config, mock_vosk_model, mock_vosk_recognizer
    ):
        """Test adding audio chunk when not listening."""
        with (
            patch("builtins.open", mock_open(read_data="")),
            patch("yaml.safe_load", return_value=mock_config),
            patch("talk2me.core.wake_word.Model", return_value=mock_vosk_model),
            patch("talk2me.core.wake_word.KaldiRecognizer", return_value=mock_vosk_recognizer),
            patch("talk2me.core.wake_word.logger") as mock_logger,
        ):
            detector = WakeWordDetector()
            detector.listening = False

            detector.add_audio_chunk(b"audio_data")

            mock_logger.warning.assert_called_once_with("Wake word detection is not running")
            assert detector.audio_queue.empty()

    def test_is_wake_word_detected(self, mock_config, mock_vosk_model, mock_vosk_recognizer):
        """Test wake word detection check."""
        with (
            patch("builtins.open", mock_open(read_data="")),
            patch("yaml.safe_load", return_value=mock_config),
            patch("talk2me.core.wake_word.Model", return_value=mock_vosk_model),
            patch("talk2me.core.wake_word.KaldiRecognizer", return_value=mock_vosk_recognizer),
        ):
            detector = WakeWordDetector()

            assert not detector.is_wake_word_detected()

            detector.detected = True
            assert detector.is_wake_word_detected()

    def test_stop_listening_success(self, mock_config, mock_vosk_model, mock_vosk_recognizer):
        """Test successful stop listening."""
        mock_thread = Mock()
        mock_thread.is_alive.return_value = True

        with (
            patch("builtins.open", mock_open(read_data="")),
            patch("yaml.safe_load", return_value=mock_config),
            patch("talk2me.core.wake_word.Model", return_value=mock_vosk_model),
            patch("talk2me.core.wake_word.KaldiRecognizer", return_value=mock_vosk_recognizer),
            patch("talk2me.core.wake_word.logger") as mock_logger,
        ):
            detector = WakeWordDetector()
            detector.listening = True
            detector.thread = mock_thread

            detector.stop_listening()

            assert not detector.listening
            assert not detector.detected
            mock_thread.join.assert_called_once_with(timeout=2.0)
            mock_logger.info.assert_called_with("Wake word detection stopped")

    def test_stop_listening_no_thread(self, mock_config, mock_vosk_model, mock_vosk_recognizer):
        """Test stop listening when no thread exists."""
        with (
            patch("builtins.open", mock_open(read_data="")),
            patch("yaml.safe_load", return_value=mock_config),
            patch("talk2me.core.wake_word.Model", return_value=mock_vosk_model),
            patch("talk2me.core.wake_word.KaldiRecognizer", return_value=mock_vosk_recognizer),
        ):
            detector = WakeWordDetector()
            detector.listening = True
            detector.thread = None

            detector.stop_listening()

            assert not detector.listening
            assert not detector.detected
