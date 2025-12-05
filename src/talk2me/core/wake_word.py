import json
import logging
import queue
import threading
from typing import cast

import numpy as np
import yaml
from vosk import KaldiRecognizer, Model

logger = logging.getLogger(__name__)


class WakeWordDetector:
    """Wake word detection engine using Vosk for offline speech recognition.

    This class loads a lightweight Vosk model and continuously processes audio chunks
    to detect configured wake word phrases. Uses threading for non-blocking audio processing.
    """

    def __init__(self, config_path: str = "config/default.yaml") -> None:
        """Initialize the wake word detector by loading the Vosk model and wake words from config.

        Args:
            config_path: Path to the configuration YAML file.

        Raises:
            FileNotFoundError: If config file or model directory is not found.
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

        if config is None:
            config = {}

        wake_word_model_path = config.get("stt", {}).get("wake_word_model_path")
        if not wake_word_model_path:
            raise ValueError("Wake word model_path not specified in configuration")

        wake_words = config.get("wake_words", {}).get("activation", [])
        if not wake_words:
            raise ValueError("Wake words not specified in configuration")

        try:
            self.model = Model(wake_word_model_path)
            logger.info(f"Successfully loaded Vosk wake word model from {wake_word_model_path}")
        except Exception as e:
            logger.error(f"Failed to load Vosk model from {wake_word_model_path}: {e}")
            raise RuntimeError(f"Model loading failed: {e}") from e

        self.sample_rate = config.get("stt", {}).get("sample_rate", 16000)
        self.wake_words = [phrase.lower() for phrase in wake_words]
        self.recognizer = KaldiRecognizer(self.model, self.sample_rate)

        self.audio_queue: queue.Queue[bytes | np.ndarray] = queue.Queue()
        self.detected = False
        self.listening = False
        self.thread: threading.Thread | None = None

    def start_listening(self) -> None:
        """Start the wake word detection thread to continuously process audio chunks.

        This method starts a background thread that monitors the audio queue and
        processes incoming audio data for wake word detection.
        """
        if self.listening:
            logger.warning("Wake word detection is already running")
            return

        self.listening = True
        self.detected = False
        self.thread = threading.Thread(target=self._process_audio, daemon=True)
        self.thread.start()
        logger.info("Wake word detection started")

    def _process_audio(self) -> None:
        """Internal method that runs in a separate thread to process audio chunks.

        Continuously checks the audio queue for new chunks, feeds them to the Vosk
        recognizer, and checks for wake word matches in both partial and final results.
        """
        while self.listening:
            try:
                audio_chunk = self.audio_queue.get(timeout=1.0)
                audio_bytes = self._convert_audio_to_bytes(audio_chunk)

                try:
                    if self.recognizer.AcceptWaveform(audio_bytes):
                        text = self._process_vosk_result(self.recognizer.Result())
                        if text and self._check_wake_word(text):
                            self.detected = True
                            logger.info(f"Wake word detected: '{text}'")
                    else:
                        partial_text = self._process_vosk_partial(self.recognizer.PartialResult())
                        if partial_text and self._check_wake_word(partial_text):
                            self.detected = True
                            logger.info(f"Wake word detected in partial: '{partial_text}'")
                except Exception as e:
                    logger.error(f"Error processing audio chunk: {e}")

            except queue.Empty:
                continue

        logger.debug("Audio processing thread stopped")

    def _check_wake_word(self, text: str) -> bool:
        """Check if the given text contains any of the configured wake words.

        Args:
            text: The recognized text to check.

        Returns:
            True if a wake word is found, False otherwise.
        """
        return any(phrase in text for phrase in self.wake_words)

    def _convert_audio_to_bytes(self, audio_chunk: bytes | np.ndarray) -> bytes:
        """Convert audio chunk to bytes format for Vosk processing.

        Args:
            audio_chunk: Audio data as bytes or numpy array.

        Returns:
            Audio data as bytes.

        Raises:
            ValueError: If audio chunk type is invalid.
        """
        if isinstance(audio_chunk, np.ndarray):
            if audio_chunk.dtype != np.int16:
                logger.warning("Converting audio data to int16")
                audio_chunk = (audio_chunk * 32767).astype(np.int16)
            return audio_chunk.tobytes()  # type: ignore
        elif isinstance(audio_chunk, bytes):
            return audio_chunk  # type: ignore
        else:
            logger.error("Invalid audio chunk type")
            raise ValueError("Invalid audio chunk type")

    def _process_vosk_result(self, result_json: str) -> str:
        """Process Vosk result JSON and extract text.

        Args:
            result_json: JSON string from Vosk.

        Returns:
            Extracted text from the result.
        """
        result = cast(dict, json.loads(result_json))
        return cast(str, result.get("text", "")).lower().strip()

    def _process_vosk_partial(self, partial_json: str) -> str:
        """Process Vosk partial result JSON and extract text.

        Args:
            partial_json: JSON string from Vosk partial result.

        Returns:
            Extracted partial text.
        """
        partial = cast(dict, json.loads(partial_json))
        return cast(str, partial.get("partial", "")).lower().strip()

    def add_audio_chunk(self, audio_chunk: bytes | np.ndarray) -> None:
        """Add an audio chunk to the processing queue.

        Args:
            audio_chunk: Audio data as bytes (PCM 16-bit mono) or numpy array (int16).
        """
        if not self.listening:
            logger.warning("Wake word detection is not running")
            return
        self.audio_queue.put(audio_chunk)

    def is_wake_word_detected(self) -> bool:
        """Check if a wake word has been detected.

        Returns:
            True if a wake word was detected, False otherwise.
        """
        return self.detected

    def stop_listening(self) -> None:
        """Stop the wake word detection thread and reset the detector state.

        This method stops the background processing thread and clears the detection flag.
        """
        self.listening = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        self.detected = False
        logger.info("Wake word detection stopped")
