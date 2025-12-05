#!/usr/bin/env python3
"""
Model Download Script for Talk2Me

Downloads required ML models (Vosk STT and XTTS v2 TTS) to the appropriate directories.
Handles network issues, permission problems, and provides progress indicators.
"""

import logging
import sys
import zipfile
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlretrieve

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Model configurations
VOSK_MODEL_URL = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
VOSK_MODEL_DIR = "vosk-model-small-en-us-0.15"
XTTS_MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
XTTS_MODEL_DIR = "xtts/v2"


class ModelDownloader:
    """Handles downloading and setup of ML models."""

    def __init__(self, config_path: str = "config/default.yaml"):
        """Initialize the downloader with configuration."""
        self.config_path = config_path
        self.models_dir = Path("models")

    def download_vosk_model(self) -> bool:
        """Download and extract Vosk STT model.

        Returns:
            True if successful, False otherwise.
        """
        model_dir = self.models_dir / VOSK_MODEL_DIR

        # Check if model already exists
        if model_dir.exists() and any(model_dir.iterdir()):
            logger.info(f"Vosk model already exists at {model_dir}")
            return True

        logger.info("Downloading Vosk STT model...")

        try:
            # Create temp file for download
            temp_zip = self.models_dir / "vosk_model_temp.zip"

            # Download with progress
            def progress_hook(block_num, block_size, total_size):
                if total_size > 0:
                    percent = min(100, (block_num * block_size * 100) / total_size)
                    print(f"\rDownloading Vosk model: {percent:.1f}%", end="", flush=True)

            urlretrieve(VOSK_MODEL_URL, temp_zip, progress_hook)
            print()  # New line after progress

            # Extract the zip
            logger.info("Extracting Vosk model...")
            with zipfile.ZipFile(temp_zip, "r") as zip_ref:
                # Get the directory name inside the zip
                zip_contents = zip_ref.namelist()
                if zip_contents:
                    # The zip contains a directory, extract to models/
                    zip_ref.extractall(self.models_dir)

            # Clean up temp file
            temp_zip.unlink()

            # Verify extraction
            if model_dir.exists() and any(model_dir.iterdir()):
                logger.info(f"Successfully downloaded Vosk model to {model_dir}")
                return True
            else:
                logger.error("Failed to extract Vosk model properly")
                return False

        except (URLError, HTTPError) as e:
            logger.error(f"Network error downloading Vosk model: {e}")
            return False
        except zipfile.BadZipFile as e:
            logger.error(f"Invalid zip file: {e}")
            return False
        except PermissionError as e:
            logger.error(f"Permission error: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error downloading Vosk model: {e}")
            return False

    def download_xtts_model(self) -> bool:
        """Download XTTS v2 TTS model using TTS library.

        Returns:
            True if successful, False otherwise.
        """
        logger.info("Checking XTTS v2 TTS model...")

        try:
            # Import TTS here to avoid import errors if not installed
            from TTS.api import TTS

            # Try to create TTS instance - this will download the model if needed
            # The TTS library handles caching automatically
            logger.info("Initializing TTS model download...")
            tts = TTS(model_name=XTTS_MODEL_NAME)

            logger.info("XTTS v2 model ready (cached by TTS library)")
            return True

        except ImportError as e:
            logger.error(f"TTS library not installed: {e}")
            logger.error("Please run 'pip install TTS' first")
            return False
        except Exception as e:
            logger.error(f"Error initializing XTTS model: {e}")
            logger.error(
                "This may be due to license agreement prompt in non-interactive environment"
            )
            logger.error("Try running the setup script in an interactive terminal")
            return False

    def verify_models(self) -> bool:
        """Verify that required models are available.

        Returns:
            True if all models are present, False otherwise.
        """
        logger.info("Verifying model availability...")

        vosk_dir = self.models_dir / VOSK_MODEL_DIR
        vosk_ok = vosk_dir.exists() and any(vosk_dir.iterdir())

        # For XTTS, check if TTS library is available (model will be downloaded when needed)
        xtts_ok = False
        try:
            import TTS

            xtts_ok = True
            logger.info("✓ XTTS v2 TTS library available (model will download on first use)")
        except ImportError:
            logger.error("✗ TTS library not available")

        if vosk_ok:
            logger.info("✓ Vosk STT model verified")
        else:
            logger.error("✗ Vosk STT model missing")

        return vosk_ok and xtts_ok


def main():
    """Main entry point for the model download script."""
    print("Talk2Me Model Downloader")
    print("=" * 40)

    downloader = ModelDownloader()

    vosk_success = downloader.download_vosk_model()
    xtts_download_success = downloader.download_xtts_model()

    # Verify all models
    verification_success = downloader.verify_models()

    # Overall success: Vosk must be downloaded, XTTS library must be available
    # XTTS model download may fail due to license prompt but that's OK if library is available
    success = vosk_success and verification_success

    if success:
        if xtts_download_success:
            print("\n✓ All models downloaded and verified successfully!")
        else:
            print("\n✓ Models verified successfully!")
            print("Note: XTTS model requires interactive license acceptance.")
            print("It will download automatically when you first run the TTS engine.")
        print("You can now run the Talk2Me application.")
        return 0
    else:
        print("\n✗ Some models failed to download. Please check the errors above.")
        print("You may need to check your internet connection or permissions.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
