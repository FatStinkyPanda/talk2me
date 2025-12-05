#!/usr/bin/env python3
"""
Talk2Me CLI - Command-line interface for the Talk2Me voice interaction system.

This module provides a command-line interface that supports:
- Interactive voice conversation mode
- API server mode
- Custom configuration file support
- Host and port configuration for API server
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any

try:
    import pyaudio

    PYAUDIO_AVAILABLE = True
except ImportError:
    pyaudio = None
    PYAUDIO_AVAILABLE = False

import yaml

from talk2me.api.main import app
from talk2me.core.wake_word import WakeWordDetector
from talk2me.stt.engine import STTEngine
from talk2me.tts.engine import TTSEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class InteractiveMode:
    """Handles interactive voice conversation mode."""

    def __init__(self, config_path: str = "config/default.yaml"):
        """Initialize interactive mode with engines and audio I/O.

        Args:
            config_path: Path to configuration file

        Raises:
            RuntimeError: If PyAudio is not available
        """
        if not PYAUDIO_AVAILABLE:
            raise RuntimeError(
                "PyAudio is required for interactive mode but is not installed. "
                "Install it with: pip install pyaudio"
            )

        self.config_path = config_path
        self.config = self._load_config()

        # Initialize engines
        self.stt_engine = STTEngine(config_path)
        self.tts_engine = TTSEngine(config_path)
        self.wake_word_detector = WakeWordDetector(config_path)

        # Audio configuration
        self.audio_config = self.config.get("audio", {})
        self.chunk_size = self.audio_config.get("chunk_size", 1024)
        self.sample_rate = self.config.get("stt", {}).get("sample_rate", 16000)

        # Audio I/O
        self.audio = pyaudio.PyAudio()
        self.input_stream = None
        self.output_stream = None

        # Control flags
        self.running = False
        self.listening = False

    def _load_config(self) -> dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path) as f:
                config = yaml.safe_load(f)
            return config or {}
        except Exception as e:
            logger.error(f"Failed to load config from {self.config_path}: {e}")
            return {}

    def _setup_audio_streams(self) -> None:
        """Set up audio input and output streams."""
        try:
            # Input stream for microphone
            self.input_stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                input_device_index=self.audio_config.get("input_device"),
            )

            # Output stream for speakers
            self.output_stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.tts_engine.sample_rate,
                output=True,
                frames_per_buffer=self.chunk_size,
                output_device_index=self.audio_config.get("output_device"),
            )

            logger.info("Audio streams initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize audio streams: {e}")
            raise

    def _play_audio(self, audio_bytes: bytes) -> None:
        """Play audio data through speakers.

        Args:
            audio_bytes: PCM audio data
        """
        if not self.output_stream:
            return

        try:
            self.output_stream.write(audio_bytes)
        except Exception as e:
            logger.error(f"Failed to play audio: {e}")

    def _record_audio_chunk(self) -> bytes | None:
        """Record a chunk of audio from microphone.

        Returns:
            Audio data as bytes, or None if recording fails
        """
        if not self.input_stream:
            return None

        try:
            return self.input_stream.read(self.chunk_size, exception_on_overflow=False)
        except Exception as e:
            logger.error(f"Failed to record audio: {e}")
            return None

    def _conversation_loop(self) -> None:
        """Main conversation loop for interactive mode."""
        print("\n🎤 Talk2Me Interactive Mode")
        print("Say your wake word to start listening...")
        print("Press Ctrl+C to exit\n")

        self.wake_word_detector.start_listening()

        try:
            while self.running:
                # Record audio chunk
                audio_chunk = self._record_audio_chunk()
                if not audio_chunk:
                    time.sleep(0.01)
                    continue

                # Add to wake word detector
                self.wake_word_detector.add_audio_chunk(audio_chunk)

                # Check for wake word
                if self.wake_word_detector.is_wake_word_detected():
                    print("🎯 Wake word detected! Listening...")

                    # Start listening for speech
                    self._handle_conversation(audio_chunk)

                    # Reset wake word detector
                    self.wake_word_detector.stop_listening()
                    self.wake_word_detector.start_listening()

                time.sleep(0.01)

        except KeyboardInterrupt:
            print("\n👋 Exiting interactive mode...")
        except Exception as e:
            logger.error(f"Error in conversation loop: {e}")
        finally:
            self.wake_word_detector.stop_listening()

    def _handle_conversation(self, initial_audio: bytes) -> None:
        """Handle a conversation turn after wake word detection.

        Args:
            initial_audio: Initial audio chunk that triggered wake word
        """
        try:
            # Collect audio until silence or done talking phrases
            audio_buffer = [initial_audio]

            # Simple silence detection (collect for a few seconds)
            silence_threshold = 2.0  # seconds
            start_time = time.time()

            while time.time() - start_time < silence_threshold and self.running:
                chunk = self._record_audio_chunk()
                if chunk:
                    audio_buffer.append(chunk)
                time.sleep(0.01)

            # Combine audio chunks
            full_audio = b"".join(audio_buffer)

            # Transcribe speech
            print("🔄 Transcribing...")
            text = self.stt_engine.transcribe(full_audio)

            if text.strip():
                print(f"👤 You said: {text}")

                # Generate response (simple echo for now)
                response_text = f"You said: {text}"
                print(f"🤖 Response: {response_text}")

                # Synthesize and play response
                print("🔊 Synthesizing response...")
                audio_response = self.tts_engine.synthesize(response_text)
                self._play_audio(audio_response)

            else:
                print("🤷 No speech detected")

        except Exception as e:
            logger.error(f"Error handling conversation: {e}")
            print("❌ Error processing your speech")

    def run(self) -> None:
        """Run the interactive mode."""
        self.running = True

        try:
            self._setup_audio_streams()
            self._conversation_loop()

        except Exception as e:
            logger.error(f"Interactive mode failed: {e}")
            print(f"❌ Interactive mode failed: {e}")
            sys.exit(1)

        finally:
            self._cleanup()

    def _cleanup(self) -> None:
        """Clean up audio resources."""
        if self.input_stream:
            self.input_stream.stop_stream()
            self.input_stream.close()

        if self.output_stream:
            self.output_stream.stop_stream()
            self.output_stream.close()

        self.audio.terminate()
        logger.info("Audio resources cleaned up")


def run_api_server(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Run the FastAPI server.

    Args:
        host: Server host
        port: Server port
    """
    import uvicorn

    print(f"🚀 Starting Talk2Me API server on {host}:{port}")
    print(f"📖 API documentation: http://{host}:{port}/docs")
    print("Press Ctrl+C to stop\n")

    try:
        uvicorn.run(app, host=host, port=port)
    except KeyboardInterrupt:
        print("\n👋 API server stopped")
    except Exception as e:
        logger.error(f"API server failed: {e}")
        print(f"❌ API server failed: {e}")
        sys.exit(1)


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="talk2me",
        description="Talk2Me - Offline voice interaction system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  talk2me                          # Start with default config
  talk2me --config myconfig.yaml   # Use custom config
  talk2me --api-only               # Run API server only
  talk2me --interactive            # Start voice conversation
  talk2me --host 127.0.0.1 --port 9000  # Custom API server settings
        """,
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="Path to configuration file (default: config/default.yaml)",
    )

    parser.add_argument(
        "--interactive", action="store_true", help="Start interactive voice conversation mode"
    )

    parser.add_argument(
        "--api-only", action="store_true", help="Run API server only (default mode)"
    )

    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="API server host (default: 0.0.0.0)"
    )

    parser.add_argument("--port", type=int, default=8000, help="API server port (default: 8000)")

    args = parser.parse_args()

    # Validate config file exists
    if not Path(args.config).exists():
        print(f"❌ Configuration file not found: {args.config}")
        sys.exit(1)

    # Determine mode
    if args.interactive:
        # Interactive mode
        print("🎤 Starting Talk2Me in interactive mode...")
        interactive = InteractiveMode(args.config)
        interactive.run()

    else:
        # API server mode (default)
        # Load config to get default host/port if not specified
        try:
            with open(args.config) as f:
                config = yaml.safe_load(f) or {}
        except Exception:
            config = {}

        api_config = config.get("api", {})
        host = args.host if args.host != "0.0.0.0" else api_config.get("host", "0.0.0.0")
        port = args.port if args.port != 8000 else api_config.get("port", 8000)

        run_api_server(host, port)


if __name__ == "__main__":
    main()
