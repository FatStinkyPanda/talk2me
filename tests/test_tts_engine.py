import os
import sys
from unittest.mock import MagicMock, Mock, mock_open, patch

import numpy as np
import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Mock TTS module before importing TTSEngine
TTS_mock = MagicMock()
with patch.dict(
    "sys.modules", {"TTS": TTS_mock, "TTS.api": TTS_mock.api, "TTS.api.TTS": TTS_mock.api.TTS}
):
    from talk2me.tts.engine import TTSEngine


@pytest.fixture
def mock_config():
    """Mock main configuration data."""
    return {
        "tts": {"model_path": "/path/to/model", "sample_rate": 24000, "default_voice": "voice1"}
    }


@pytest.fixture
def mock_voices_config():
    """Mock voices configuration data."""
    return {
        "voices": {
            "voice1": {"name": "Voice One", "language": "en", "samples_dir": "/path/to/samples1"},
            "voice2": {"name": "Voice Two", "language": "es", "samples_dir": "/path/to/samples2"},
        }
    }


@pytest.fixture
def mock_tts_model():
    """Mock TTS model."""
    return Mock()


class TestTTSEngine:
    """Test suite for TTSEngine class."""

    @patch("talk2me.tts.engine.TTS")
    def test_init_success(self, mock_tts_class, mock_config, mock_voices_config, mock_tts_model):
        """Test successful initialization."""
        mock_tts_class.return_value = mock_tts_model

        with (
            patch("builtins.open", mock_open()),
            patch("yaml.safe_load", side_effect=[mock_config, mock_voices_config]),
            patch("pathlib.Path") as mock_path,
        ):
            # Mock path exists
            mock_path.return_value.exists.return_value = True

            engine = TTSEngine()

            assert engine.sample_rate == 24000
            assert engine.default_voice == "voice1"
            assert engine.voices == mock_voices_config["voices"]
            assert engine.tts == mock_tts_model

    def test_load_config_file_not_found(self):
        """Test config loading when file is not found."""
        with (
            patch("builtins.open", side_effect=FileNotFoundError),
            pytest.raises(FileNotFoundError),
        ):
            TTSEngine("nonexistent.yaml")

    def test_load_config_yaml_error(self):
        """Test config loading when YAML parsing fails."""
        with (
            patch("builtins.open", mock_open(read_data="invalid: yaml: [")),
            patch("yaml.safe_load", side_effect=yaml.YAMLError("YAML error")),
            pytest.raises(RuntimeError, match="Invalid configuration file"),
        ):
            TTSEngine()

    def test_load_voices_config_file_not_found(self, mock_config):
        """Test voices config loading when file is not found."""
        with (
            patch("builtins.open", mock_open()),
            patch("yaml.safe_load", side_effect=[mock_config, Exception("File not found")]),
            pytest.raises(FileNotFoundError),
        ):
            TTSEngine()

    def test_load_voices_config_yaml_error(self, mock_config):
        """Test voices config loading when YAML parsing fails."""
        with (
            patch("builtins.open", mock_open()),
            patch("yaml.safe_load", side_effect=[mock_config, yaml.YAMLError("YAML error")]),
            pytest.raises(RuntimeError, match="Invalid voices configuration file"),
        ):
            TTSEngine()

    def test_initialize_attributes_missing_model_path(self):
        """Test initialization with missing model path."""
        config = {"tts": {}}
        voices_config = {"voices": {"voice1": {"samples_dir": "/path"}}}

        with (
            patch("builtins.open", mock_open()),
            patch("yaml.safe_load", side_effect=[config, voices_config]),
            pytest.raises(ValueError, match="TTS model_path not specified"),
        ):
            TTSEngine()

    def test_initialize_attributes_missing_voices(self, mock_config):
        """Test initialization with missing voices."""
        voices_config = {}

        with (
            patch("builtins.open", mock_open()),
            patch("yaml.safe_load", side_effect=[mock_config, voices_config]),
            pytest.raises(ValueError, match="No voices configured"),
        ):
            TTSEngine()

    def test_validate_voices_missing_samples_dir(self, mock_config):
        """Test voice validation with missing samples directory."""
        voices_config = {
            "voices": {
                "voice1": {"name": "Voice One"}  # Missing samples_dir
            }
        }

        with (
            patch("builtins.open", mock_open()),
            patch("yaml.safe_load", side_effect=[mock_config, voices_config]),
            pytest.raises(ValueError, match="Voice 'voice1' missing samples_dir"),
        ):
            TTSEngine()

    @patch("talk2me.tts.engine.TTS")
    def test_validate_voices_samples_dir_not_exists(
        self, _mock_tts_class, mock_config, mock_voices_config
    ):
        """Test voice validation when samples directory doesn't exist."""
        with (
            patch("builtins.open", mock_open()),
            patch("yaml.safe_load", side_effect=[mock_config, mock_voices_config]),
            patch("pathlib.Path") as mock_path,
            patch("talk2me.tts.engine.logger") as mock_logger,
        ):
            # Mock path doesn't exist
            mock_path_instance = Mock()
            mock_path_instance.exists.return_value = False
            mock_path.return_value = mock_path_instance

            TTSEngine()

            mock_logger.warning.assert_called()

    @patch("talk2me.tts.engine.TTS", side_effect=Exception("Model load failed"))
    def test_load_model_failure(self, _mock_tts_class, mock_config, mock_voices_config):
        """Test model loading failure."""
        with (
            patch("builtins.open", mock_open()),
            patch("yaml.safe_load", side_effect=[mock_config, mock_voices_config]),
            patch("pathlib.Path") as mock_path,
        ):
            mock_path.return_value.exists.return_value = True

            with pytest.raises(RuntimeError, match="Model loading failed"):
                TTSEngine()

    @patch("talk2me.tts.engine.TTS")
    def test_synthesize_success(
        self, mock_tts_class, mock_config, mock_voices_config, mock_tts_model
    ):
        """Test successful synthesis."""
        mock_tts_class.return_value = mock_tts_model
        # Mock TTS output
        mock_audio = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        mock_tts_model.tts.return_value = mock_audio

        with (
            patch("builtins.open", mock_open()),
            patch("yaml.safe_load", side_effect=[mock_config, mock_voices_config]),
            patch("pathlib.Path") as mock_path,
            patch("os.path.exists", return_value=True),
            patch("os.listdir", return_value=["sample.wav"]),
        ):
            mock_path.return_value.exists.return_value = True
            # Mock glob to return sample files
            mock_path.return_value.glob.return_value = [Mock()]
            mock_path.return_value.__str__ = Mock(return_value="/path/to/sample.wav")

            engine = TTSEngine()

            result = engine.synthesize("Hello world", "voice1")

            assert isinstance(result, bytes)
            mock_tts_model.tts.assert_called_once()

    def test_synthesize_default_voice(self, mock_config, mock_voices_config, mock_tts_model):
        """Test synthesis with default voice."""
        mock_audio = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        mock_tts_model.tts.return_value = mock_audio

        with (
            patch("builtins.open", mock_open()),
            patch("yaml.safe_load", side_effect=[mock_config, mock_voices_config]),
            patch("pathlib.Path") as mock_path,
            patch("talk2me.tts.engine.TTS", return_value=mock_tts_model),
            patch("os.path.exists", return_value=True),
            patch("os.listdir", return_value=["sample.wav"]),
        ):
            mock_path.return_value.exists.return_value = True
            mock_path.return_value.glob.return_value = [Mock()]
            mock_path.return_value.__str__ = Mock(return_value="/path/to/sample.wav")

            engine = TTSEngine()

            result = engine.synthesize("Hello world")  # No voice specified

            assert isinstance(result, bytes)
            # Should use default voice
            mock_tts_model.tts.assert_called_once()

    def test_synthesize_invalid_voice(self, mock_config, mock_voices_config, mock_tts_model):
        """Test synthesis with invalid voice."""
        with (
            patch("builtins.open", mock_open()),
            patch("yaml.safe_load", side_effect=[mock_config, mock_voices_config]),
            patch("pathlib.Path") as mock_path,
            patch("talk2me.tts.engine.TTS", return_value=mock_tts_model),
        ):
            mock_path.return_value.exists.return_value = True

            engine = TTSEngine()

            with pytest.raises(ValueError, match="Voice 'invalid' not configured"):
                engine.synthesize("Hello", "invalid")

    def test_synthesize_no_sample_files(self, mock_config, mock_voices_config, mock_tts_model):
        """Test synthesis when no sample files found."""
        with (
            patch("builtins.open", mock_open()),
            patch("yaml.safe_load", side_effect=[mock_config, mock_voices_config]),
            patch("pathlib.Path") as mock_path,
            patch("talk2me.tts.engine.TTS", return_value=mock_tts_model),
            patch("os.path.exists", return_value=True),
            patch("os.listdir", return_value=[]),
        ):
            mock_path.return_value.exists.return_value = True
            mock_path.return_value.glob.return_value = []  # No sample files

            engine = TTSEngine()

            with pytest.raises(ValueError, match="No WAV sample files found"):
                engine.synthesize("Hello", "voice1")

    def test_synthesize_tts_error(self, mock_config, mock_voices_config, mock_tts_model):
        """Test synthesis when TTS processing fails."""
        mock_tts_model.tts.side_effect = Exception("TTS error")

        with (
            patch("builtins.open", mock_open()),
            patch("yaml.safe_load", side_effect=[mock_config, mock_voices_config]),
            patch("pathlib.Path") as mock_path,
            patch("talk2me.tts.engine.TTS", return_value=mock_tts_model),
            patch("os.path.exists", return_value=True),
            patch("os.listdir", return_value=["sample.wav"]),
        ):
            mock_path.return_value.exists.return_value = True
            mock_path.return_value.glob.return_value = [Mock()]
            mock_path.return_value.__str__ = Mock(return_value="/path/to/sample.wav")

            engine = TTSEngine()

            with pytest.raises(RuntimeError, match="Synthesis failed"):
                engine.synthesize("Hello world")

    def test_synthesize_audio_conversion(self, mock_config, mock_voices_config, mock_tts_model):
        """Test audio data type conversion during synthesis."""
        # Mock TTS to return different data types
        mock_audio = np.array([0.1, 0.2, 0.3], dtype=np.float64)  # Not float32
        mock_tts_model.tts.return_value = mock_audio

        with (
            patch("builtins.open", mock_open()),
            patch("yaml.safe_load", side_effect=[mock_config, mock_voices_config]),
            patch("pathlib.Path") as mock_path,
            patch("talk2me.tts.engine.TTS", return_value=mock_tts_model),
            patch("os.path.exists", return_value=True),
            patch("os.listdir", return_value=["sample.wav"]),
        ):
            mock_path.return_value.exists.return_value = True
            mock_path.return_value.glob.return_value = [Mock()]
            mock_path.return_value.__str__ = Mock(return_value="/path/to/sample.wav")

            engine = TTSEngine()

            result = engine.synthesize("Hello world")

            assert isinstance(result, bytes)
            # Should convert to int16 PCM
            mock_tts_model.tts.assert_called_once()

    def test_synthesize_numpy_array_handling(self, mock_config, mock_voices_config, mock_tts_model):
        """Test handling of non-numpy TTS output."""
        # Mock TTS to return list instead of numpy array
        mock_tts_model.tts.return_value = [0.1, 0.2, 0.3]

        with (
            patch("builtins.open", mock_open()),
            patch("yaml.safe_load", side_effect=[mock_config, mock_voices_config]),
            patch("pathlib.Path") as mock_path,
            patch("talk2me.tts.engine.TTS", return_value=mock_tts_model),
            patch("os.path.exists", return_value=True),
            patch("os.listdir", return_value=["sample.wav"]),
        ):
            mock_path.return_value.exists.return_value = True
            mock_path.return_value.glob.return_value = [Mock()]
            mock_path.return_value.__str__ = Mock(return_value="/path/to/sample.wav")

            engine = TTSEngine()

            result = engine.synthesize("Hello world")

            assert isinstance(result, bytes)
