import json
import logging

import numpy as np
import yaml
from vosk import KaldiRecognizer, Model

logger = logging.getLogger(__name__)


class STTEngine:
    """
    Speech-to-Text engine using Vosk for offline speech recognition.

    This class loads a Vosk model and provides transcription functionality
    for audio data in bytes or numpy array format.
    """

    def __init__(self, config_path: str = "config/default.yaml") -> None:
        """
        Initialize the STT engine by loading the Vosk model from config.

        Args:
            config_path: Path to the configuration YAML file.

        Raises:
            FileNotFoundError: If config file or model directory is not found.
            RuntimeError: If model loading fails.
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

        model_path = config.get("stt", {}).get("model_path")
        if not model_path:
            raise ValueError("STT model_path not specified in configuration")

        try:
            self.model = Model(model_path)
            logger.info(f"Successfully loaded Vosk model from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load Vosk model from {model_path}: {e}")
            raise RuntimeError(f"Model loading failed: {e}") from e

        self.sample_rate = config.get("stt", {}).get("sample_rate", 16000)

    def transcribe(self, audio_data: bytes | np.ndarray, sample_rate: int | None = None) -> str:
        """
        Transcribe audio data to text using the loaded Vosk model.

        Args:
            audio_data: Audio data as bytes (PCM 16-bit mono) or numpy array (int16).
            sample_rate: Sample rate of the audio. If None, uses config default.

        Returns:
            Transcribed text as a string.

        Raises:
            ValueError: If audio data format is invalid.
            RuntimeError: If transcription fails.
        """
        if sample_rate is None:
            sample_rate = self.sample_rate

        # Convert numpy array to bytes if necessary
        if isinstance(audio_data, np.ndarray):
            if audio_data.dtype != np.int16:
                logger.warning("Converting audio data to int16")
                audio_data = (audio_data * 32767).astype(np.int16)
            audio_bytes = audio_data.tobytes()
        elif isinstance(audio_data, bytes):
            audio_bytes = audio_data
        else:
            raise ValueError("Audio data must be bytes or numpy array")

        try:
            recognizer = KaldiRecognizer(self.model, sample_rate)
            recognizer.SetWords(True)  # Include word timestamps if needed

            results = []
            chunk_size = 4000  # Process in chunks

            for i in range(0, len(audio_bytes), chunk_size * 2):  # *2 for 16-bit
                chunk = audio_bytes[i : i + chunk_size * 2]
                if recognizer.AcceptWaveform(chunk):
                    result = recognizer.Result()
                    # Parse JSON result
                    result_dict = json.loads(result)
                    if "text" in result_dict:
                        results.append(result_dict["text"])
                else:
                    # Partial result, could collect if needed
                    pass

            # Final result
            final_result = recognizer.FinalResult()
            final_dict = json.loads(final_result)
            if "text" in final_dict:
                results.append(final_dict["text"])

            transcribed_text = " ".join(results).strip()
            logger.debug(f"Transcription completed: {len(transcribed_text)} characters")
            return transcribed_text

        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise RuntimeError(f"Transcription failed: {e}") from e
