import logging
import os

# Fix for pathlib compatibility issue in TTS library
import pathlib
from typing import Any, cast

import numpy as np
import yaml
from TTS.api import TTS

pathlib.Path._flavour = pathlib.Path()._flavour

logger = logging.getLogger(__name__)


class TTSEngine:
    """Text-to-Speech engine using XTTS v2 model for multilingual speech synthesis.

    This class loads the XTTS v2 model and provides text-to-speech functionality
    with support for different voices configured in voices.yaml.
    """

    def __init__(
        self, config_path: str = "config/default.yaml", voices_path: str = "config/voices.yaml"
    ) -> None:
        """Initialize the TTS engine by loading the XTTS v2 model and voice configurations.

        Args:
            config_path: Path to the main configuration YAML file.
            voices_path: Path to the voices configuration YAML file.

        Raises:
            FileNotFoundError: If config files or model directory is not found.
            RuntimeError: If model loading fails.
            ValueError: If required configuration is missing.
        """
        config = self._load_config(config_path)
        voices_config = self._load_voices_config(voices_path)

        self._initialize_attributes(config, voices_config)
        self._validate_voices()
        self._load_model()

    def _load_config(self, config_path: str) -> dict[Any, Any]:
        """Load the main configuration from YAML file.

        Args:
            config_path: Path to the configuration file.

        Returns:
            Loaded configuration dictionary.

        Raises:
            FileNotFoundError: If config file is not found.
            RuntimeError: If YAML parsing fails.
        """
        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing configuration file: {e}")
            raise RuntimeError(f"Invalid configuration file: {e}") from e

        if config is None:
            config = {}
        return config

    def _load_voices_config(self, voices_path: str) -> dict[Any, Any]:
        """Load the voices configuration from YAML file.

        Args:
            voices_path: Path to the voices configuration file.

        Returns:
            Loaded voices configuration dictionary.

        Raises:
            FileNotFoundError: If voices config file is not found.
            RuntimeError: If YAML parsing fails.
        """
        try:
            with open(voices_path) as f:
                voices_config = yaml.safe_load(f)
        except FileNotFoundError:
            logger.error(f"Voices configuration file not found: {voices_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing voices configuration file: {e}")
            raise RuntimeError(f"Invalid voices configuration file: {e}") from e

        if voices_config is None:
            voices_config = {}
        return voices_config

    def _initialize_attributes(self, config: dict, voices_config: dict) -> None:
        """Initialize instance attributes from configurations.

        Args:
            config: Main configuration dictionary.
            voices_config: Voices configuration dictionary.

        Raises:
            ValueError: If required configuration is missing.
        """
        tts_config = config.get("tts", {})
        model_path = tts_config.get("model_path")
        if not model_path:
            raise ValueError("TTS model_path not specified in configuration")

        self.sample_rate = tts_config.get("sample_rate", 24000)
        self.default_voice = tts_config.get("default_voice", "default")
        self.voices = voices_config.get("voices", {})
        if not self.voices:
            raise ValueError("No voices configured in voices.yaml")

    def _validate_voices(self) -> None:
        """Validate voice configurations.

        Raises:
            ValueError: If voice configuration is invalid.
        """
        for voice_name, voice_info in self.voices.items():
            samples_dir = voice_info.get("samples_dir")
            if not samples_dir:
                raise ValueError(f"Voice '{voice_name}' missing samples_dir")
            if not os.path.exists(samples_dir):
                logger.warning(
                    f"Samples directory for voice '{voice_name}' does not exist: {samples_dir}"
                )

    def _load_model(self) -> None:
        """Load the XTTS v2 model.

        Raises:
            RuntimeError: If model loading fails.
        """
        try:
            self.tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")
            logger.info("Successfully loaded XTTS v2 model")
        except Exception as e:
            logger.error(f"Failed to load XTTS v2 model: {e}")
            raise RuntimeError(f"Model loading failed: {e}") from e

    def synthesize(self, text: str, voice_name: str | None = None) -> bytes:
        """Synthesize text to speech audio.

        Args:
            text: The text to synthesize.
            voice_name: Name of the voice to use. If None, uses default voice.

        Returns:
            Synthesized audio as bytes (PCM 16-bit mono).

        Raises:
            ValueError: If voice is not configured or no sample files found.
            RuntimeError: If synthesis fails.
        """
        if voice_name is None:
            voice_name = self.default_voice

        if voice_name not in self.voices:
            raise ValueError(f"Voice '{voice_name}' not configured")

        voice_info = self.voices[voice_name]
        samples_dir = voice_info["samples_dir"]
        language = voice_info.get("language", "en")

        # Find all sample WAV files
        if not os.path.exists(samples_dir):
            raise ValueError(f"Samples directory does not exist: {samples_dir}")
        sample_files = [
            os.path.join(samples_dir, f) for f in os.listdir(samples_dir) if f.endswith(".wav")
        ]
        if not sample_files:
            raise ValueError(f"No WAV sample files found in {samples_dir}")

        # Use all available sample files for better voice cloning quality
        speaker_wav = sample_files

        try:
            # Synthesize audio
            wav = cast(
                np.ndarray, self.tts.tts(text=text, speaker_wav=speaker_wav, language=language)
            )

            # Convert to numpy array if not already
            if not isinstance(wav, np.ndarray):
                wav = np.array(wav)

            # Ensure it's float32, then convert to int16
            if wav.dtype != np.float32:
                wav = wav.astype(np.float32)

            # Normalize to [-1, 1] if needed (XTTS should already be normalized)
            wav = np.clip(wav, -1.0, 1.0)

            # Convert to int16 PCM
            pcm_data = (wav * 32767).astype(np.int16)

            # Convert to bytes
            audio_bytes = pcm_data.tobytes()

            logger.debug(f"Synthesis completed: {len(audio_bytes)} bytes")
            return audio_bytes  # type: ignore

        except Exception as e:
            logger.error(f"Synthesis failed for text '{text[:50]}...': {e}")
            raise RuntimeError(f"Synthesis failed: {e}") from e
