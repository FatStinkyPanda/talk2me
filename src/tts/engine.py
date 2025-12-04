import logging
from pathlib import Path
from typing import Optional

import numpy as np
import yaml
from TTS.api import TTS

logger = logging.getLogger(__name__)


class TTSEngine:
    """
    Text-to-Speech engine using XTTS v2 model for multilingual speech synthesis.

    This class loads the XTTS v2 model and provides text-to-speech functionality
    with support for different voices configured in voices.yaml.
    """

    def __init__(self, config_path: str = "config/default.yaml", voices_path: str = "config/voices.yaml") -> None:
        """
        Initialize the TTS engine by loading the XTTS v2 model and voice configurations.

        Args:
            config_path: Path to the main configuration YAML file.
            voices_path: Path to the voices configuration YAML file.

        Raises:
            FileNotFoundError: If config files or model directory is not found.
            RuntimeError: If model loading fails.
            ValueError: If required configuration is missing.
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

        try:
            with open(voices_path) as f:
                voices_config = yaml.safe_load(f)
        except FileNotFoundError:
            logger.error(f"Voices configuration file not found: {voices_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing voices configuration file: {e}")
            raise RuntimeError(f"Invalid voices configuration file: {e}") from e

        tts_config = config.get("tts", {})
        model_path = tts_config.get("model_path")
        if not model_path:
            raise ValueError("TTS model_path not specified in configuration")

        self.sample_rate = tts_config.get("sample_rate", 24000)
        self.default_voice = tts_config.get("default_voice", "default")

        # Load voices
        self.voices = voices_config.get("voices", {})
        if not self.voices:
            raise ValueError("No voices configured in voices.yaml")

        # Validate voice configurations
        for voice_name, voice_info in self.voices.items():
            samples_dir = voice_info.get("samples_dir")
            if not samples_dir:
                raise ValueError(f"Voice '{voice_name}' missing samples_dir")
            if not Path(samples_dir).exists():
                logger.warning(f"Samples directory for voice '{voice_name}' does not exist: {samples_dir}")

        try:
            # Load XTTS v2 model
            self.tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")
            logger.info(f"Successfully loaded XTTS v2 model")
        except Exception as e:
            logger.error(f"Failed to load XTTS v2 model: {e}")
            raise RuntimeError(f"Model loading failed: {e}") from e

    def synthesize(self, text: str, voice_name: Optional[str] = None) -> bytes:
        """
        Synthesize text to speech audio.

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
        samples_dir = Path(voice_info["samples_dir"])
        language = voice_info.get("language", "en")

        # Find a sample WAV file
        sample_files = list(samples_dir.glob("*.wav"))
        if not sample_files:
            raise ValueError(f"No WAV sample files found in {samples_dir}")

        # Use the first sample file
        speaker_wav = str(sample_files[0])

        try:
            # Synthesize audio
            wav = self.tts.tts(text=text, speaker_wav=speaker_wav, language=language)

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
            return audio_bytes

        except Exception as e:
            logger.error(f"Synthesis failed for text '{text[:50]}...': {e}")
            raise RuntimeError(f"Synthesis failed: {e}") from e