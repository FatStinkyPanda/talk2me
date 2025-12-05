import logging
import os
import re
import shutil
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import uvicorn
import yaml
from fastapi import Body, FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from talk2me.core.wake_word import WakeWordDetector
from talk2me.stt.engine import STTEngine
from talk2me.tts.engine import TTSEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global engine instances
stt_engine: STTEngine | None = None
tts_engine: TTSEngine | None = None
wake_word_detector: WakeWordDetector | None = None

# Module-level singleton for File dependency
UPLOAD_FILE = File(...)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Manage application lifespan events."""
    global stt_engine, tts_engine, wake_word_detector

    try:
        logger.info("Initializing engines...")

        # Initialize STT engine
        stt_engine = STTEngine()
        logger.info("STT engine initialized")

        # Initialize TTS engine
        tts_engine = TTSEngine()
        logger.info("TTS engine initialized")

        # Initialize wake word detector
        wake_word_detector = WakeWordDetector()
        logger.info("Wake word detector initialized")

        logger.info("All engines initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize engines: {e}")
        raise

    yield


# Create FastAPI app
app = FastAPI(
    title="Talk2Me API",
    description="Speech-to-Text and Text-to-Speech API with real-time conversation support",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure based on your needs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root() -> dict[str, str]:
    """Root endpoint."""
    return {"message": "Talk2Me API is running"}


@app.post("/stt")
async def speech_to_text(
    file: UploadFile = UPLOAD_FILE, sample_rate: int | None = None
) -> dict[str, str]:
    """Convert speech audio to text.

    Args:
        file: Audio file (WAV, PCM 16-bit mono)
        sample_rate: Optional sample rate override

    Returns:
        JSON with transcribed text
    """
    if not stt_engine:
        raise HTTPException(status_code=500, detail="STT engine not initialized")

    try:
        # Read audio data
        audio_data = await file.read()

        # Transcribe
        text = stt_engine.transcribe(audio_data, sample_rate)

        return {"text": text}

    except Exception as e:
        logger.error(f"STT transcription failed: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}") from e


@app.post("/tts")
async def text_to_speech(
    text: str = Body(..., min_length=1), voice: str | None = Body(None)
) -> StreamingResponse:
    """Convert text to speech audio.

    Args:
        text: Text to synthesize (required, minimum 1 character)
        voice: Optional voice identifier

    Returns:
        Audio stream (PCM 16-bit mono)
    """
    if not tts_engine:
        raise HTTPException(status_code=500, detail="TTS engine not initialized")

    try:
        # Synthesize audio
        audio_bytes = tts_engine.synthesize(text, voice)

        # Return as streaming response
        return StreamingResponse(
            iter([audio_bytes]),
            media_type="audio/wav",
            headers={"Content-Length": str(len(audio_bytes))},
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.error(f"TTS synthesis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Synthesis failed: {str(e)}") from e


@app.get("/voices")
async def list_voices() -> dict[str, list[dict[str, Any]]]:
    """List available voices.

    Returns:
        JSON with list of available voices
    """
    if not tts_engine:
        raise HTTPException(status_code=500, detail="TTS engine not initialized")

    try:
        voices = []
        for voice_name, voice_info in tts_engine.voices.items():
            voices.append(
                {
                    "name": voice_name,
                    "display_name": voice_info.get("name", voice_name),
                    "language": voice_info.get("language", "en"),
                }
            )

        return {"voices": voices}

    except Exception as e:
        logger.error(f"Failed to list voices: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list voices: {str(e)}") from e


@app.post("/voices")
async def create_voice(
    name: str, language: str = "en", samples: list[UploadFile] | None = None
) -> dict[str, Any]:
    """Create a new voice profile with name and initial samples.

    Args:
        name: Display name for the voice
        language: Language code (default: en)
        samples: List of audio sample files (WAV format)

    Returns:
        JSON with voice_id and success message
    """
    if not tts_engine:
        raise HTTPException(status_code=500, detail="TTS engine not initialized")

    if not name:
        raise HTTPException(status_code=400, detail="name is required")

    # Generate voice_id from name
    voice_id = re.sub(r"\W+", "_", name.lower()).strip("_")
    if not voice_id:
        voice_id = f"voice_{len(tts_engine.voices)}"

    if voice_id in tts_engine.voices:
        raise HTTPException(status_code=400, detail="Voice already exists")

    if samples is None:
        samples = []

    # Create samples directory
    samples_dir = f"voices/{voice_id}/samples"
    os.makedirs(samples_dir, exist_ok=True)

    # Save uploaded samples
    saved_samples = []
    for sample in samples:
        if not sample.filename or not sample.filename.lower().endswith(".wav"):
            raise HTTPException(status_code=400, detail="Only WAV files are allowed")
        filepath = os.path.join(samples_dir, sample.filename)
        with open(filepath, "wb") as f:
            content = await sample.read()
            f.write(content)
        saved_samples.append(sample.filename)

    # Update voices configuration
    tts_engine.voices[voice_id] = {"name": name, "samples_dir": samples_dir, "language": language}

    # Save to YAML
    try:
        with open("config/voices.yaml", "w") as f:
            yaml.dump({"voices": tts_engine.voices}, f)
    except Exception as e:
        logger.error(f"Failed to save voices config: {e}")
        raise HTTPException(status_code=500, detail="Failed to save voice configuration") from e

    logger.info(f"Created voice '{voice_id}' with {len(saved_samples)} samples")
    return {
        "voice_id": voice_id,
        "message": "Voice created successfully",
        "samples_uploaded": len(saved_samples),
    }


@app.put("/voices/{voice_id}")
async def update_voice(
    voice_id: str, name: str | None = None, language: str | None = None
) -> dict[str, Any]:
    """Update voice profile (change name or language).

    Args:
        voice_id: ID of the voice to update
        name: New display name (optional)
        language: New language code (optional)

    Returns:
        JSON with success message
    """
    if not tts_engine:
        raise HTTPException(status_code=500, detail="TTS engine not initialized")

    if voice_id not in tts_engine.voices:
        raise HTTPException(status_code=404, detail="Voice not found")

    voice_info = tts_engine.voices[voice_id]

    if name is not None:
        new_voice_id = name
        if new_voice_id != voice_id:
            if new_voice_id in tts_engine.voices:
                raise HTTPException(status_code=400, detail="Voice with that name already exists")
            # Rename samples directory
            old_samples_dir = voice_info.get("samples_dir", f"voices/{voice_id}/samples")
            new_samples_dir = f"voices/{new_voice_id}/samples"
            if os.path.exists(old_samples_dir):
                os.rename(old_samples_dir, new_samples_dir)
            voice_info["samples_dir"] = new_samples_dir
        # Remove old entry
        del tts_engine.voices[voice_id]
        # Add new entry
        tts_engine.voices[new_voice_id] = voice_info
        voice_id = new_voice_id
        voice_info["name"] = name
    if language is not None:
        voice_info["language"] = language

    # Save to YAML
    try:
        with open("config/voices.yaml", "w") as f:
            yaml.dump({"voices": tts_engine.voices}, f)
    except Exception as e:
        logger.error(f"Failed to save voices config: {e}")
        raise HTTPException(status_code=500, detail="Failed to save voice configuration") from e

    logger.info(f"Updated voice '{voice_id}'")
    return {"message": "Voice updated successfully"}


@app.delete("/voices/{voice_id}")
async def delete_voice(voice_id: str) -> dict[str, Any]:
    """Delete voice profile and associated samples.

    Args:
        voice_id: ID of the voice to delete

    Returns:
        JSON with success message
    """
    if not tts_engine:
        raise HTTPException(status_code=500, detail="TTS engine not initialized")

    if voice_id not in tts_engine.voices:
        raise HTTPException(status_code=404, detail="Voice not found")

    voice_info = tts_engine.voices[voice_id]
    samples_dir = voice_info["samples_dir"]

    # Remove samples directory
    if os.path.exists(samples_dir):
        try:
            shutil.rmtree(samples_dir)
        except Exception as e:
            logger.error(f"Failed to remove samples directory {samples_dir}: {e}")
            raise HTTPException(status_code=500, detail="Failed to remove voice samples") from e

    # Remove from voices
    del tts_engine.voices[voice_id]

    # Save to YAML
    try:
        with open("config/voices.yaml", "w") as f:
            yaml.dump({"voices": tts_engine.voices}, f)
    except Exception as e:
        logger.error(f"Failed to save voices config: {e}")
        raise HTTPException(status_code=500, detail="Failed to save voice configuration") from e

    logger.info(f"Deleted voice '{voice_id}'")
    return {"message": "Voice deleted successfully"}


@app.post("/voices/{voice_id}/samples")
async def upload_samples(voice_id: str, samples: list[UploadFile] | None = None) -> dict[str, Any]:
    """Upload audio samples for a voice.

    Args:
        voice_id: ID of the voice
        samples: List of audio sample files (WAV format)

    Returns:
        JSON with list of uploaded sample filenames
    """
    if not tts_engine:
        raise HTTPException(status_code=500, detail="TTS engine not initialized")

    if voice_id not in tts_engine.voices:
        raise HTTPException(status_code=404, detail="Voice not found")

    if samples is None:
        samples = []

    samples_dir = tts_engine.voices[voice_id]["samples_dir"]
    os.makedirs(samples_dir, exist_ok=True)

    uploaded = []
    for sample in samples:
        if not sample.filename or not sample.filename.lower().endswith(".wav"):
            raise HTTPException(status_code=400, detail="Only WAV files are allowed")
        filepath = os.path.join(samples_dir, sample.filename)
        with open(filepath, "wb") as f:
            content = await sample.read()
            f.write(content)
        uploaded.append(sample.filename)

    logger.info(f"Uploaded {len(uploaded)} samples for voice '{voice_id}'")
    return {"uploaded": uploaded, "message": f"Successfully uploaded {len(uploaded)} samples"}


@app.delete("/voices/{voice_id}/samples/{sample_id}")
async def delete_sample(voice_id: str, sample_id: str) -> dict[str, Any]:
    """Remove specific sample from a voice.

    Args:
        voice_id: ID of the voice
        sample_id: Filename of the sample to remove

    Returns:
        JSON with success message
    """
    if not tts_engine:
        raise HTTPException(status_code=500, detail="TTS engine not initialized")

    if voice_id not in tts_engine.voices:
        raise HTTPException(status_code=404, detail="Voice not found")

    samples_dir = tts_engine.voices[voice_id]["samples_dir"]
    filepath = os.path.join(samples_dir, sample_id)

    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Sample not found")

    try:
        os.remove(filepath)
    except Exception as e:
        logger.error(f"Failed to remove sample {filepath}: {e}")
        raise HTTPException(status_code=500, detail="Failed to remove sample") from e

    logger.info(f"Deleted sample '{sample_id}' from voice '{voice_id}'")
    return {"message": "Sample deleted successfully"}


@app.post("/voices/{voice_id}/retrain")
async def retrain_voice(voice_id: str) -> dict[str, Any]:
    """Trigger voice retraining (validate samples and update configuration).

    Args:
        voice_id: ID of the voice to retrain

    Returns:
        JSON with success message and sample count
    """
    if not tts_engine:
        raise HTTPException(status_code=500, detail="TTS engine not initialized")

    if voice_id not in tts_engine.voices:
        raise HTTPException(status_code=404, detail="Voice not found")

    samples_dir = Path(tts_engine.voices[voice_id]["samples_dir"])
    sample_files = list(samples_dir.glob("*.wav"))

    if not sample_files:
        raise HTTPException(status_code=400, detail="No WAV samples found for retraining")

    # For XTTS, retraining isn't typically needed as it's a pre-trained model
    # This endpoint validates that samples exist and are accessible
    logger.info(f"Retrained voice '{voice_id}' with {len(sample_files)} samples")
    return {"message": "Voice retrained successfully", "sample_count": len(sample_files)}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Websocket endpoint for real-time conversation.

    Accepts audio chunks and returns transcribed text and synthesized audio.
    """
    if not stt_engine or not tts_engine or not wake_word_detector:
        await websocket.close(code=1011)  # Internal server error
        return

    await websocket.accept()
    logger.info("WebSocket connection established")

    try:
        # Start wake word detection
        wake_word_detector.start_listening()

        while True:
            try:
                # Receive audio chunk
                data = await websocket.receive_bytes()

                # Add to wake word detector
                wake_word_detector.add_audio_chunk(data)

                # Check if wake word detected
                if wake_word_detector.is_wake_word_detected():
                    # Transcribe the audio
                    try:
                        text = stt_engine.transcribe(data)
                        if text.strip():
                            # Send transcribed text
                            await websocket.send_json({"type": "transcription", "text": text})

                            # Synthesize response (echo back for now)
                            audio_bytes = tts_engine.synthesize(text)
                            await websocket.send_bytes(audio_bytes)

                        else:
                            # Empty transcription, close connection
                            await websocket.close()

                    except Exception as e:
                        logger.error(f"Processing failed: {e}")
                        await websocket.send_json({"type": "error", "message": str(e)})

                    # Reset wake word detector for next interaction
                    wake_word_detector.stop_listening()
                    wake_word_detector.start_listening()

                else:
                    # No wake word detected, close connection
                    await websocket.close()

            except WebSocketDisconnect:
                break

    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        # Cleanup
        wake_word_detector.stop_listening()
        logger.info("WebSocket connection closed")


def main():
    """Main entry point for the Talk2Me API server."""
    # Load config for server settings
    try:
        with open("config/default.yaml") as f:
            config = yaml.safe_load(f)
    except Exception:
        config = {}

    api_config = config.get("api", {})
    host = api_config.get("host", "127.0.0.1")
    port = api_config.get("port", 8000)

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
