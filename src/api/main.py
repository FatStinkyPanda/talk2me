import logging
from typing import Any

import uvicorn
import yaml
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from src.core.wake_word import WakeWordDetector
from src.stt.engine import STTEngine
from src.tts.engine import TTSEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global engine instances
stt_engine: STTEngine | None = None
tts_engine: TTSEngine | None = None
wake_word_detector: WakeWordDetector | None = None

# Module-level singleton for File dependency
UPLOAD_FILE = File(...)

# Create FastAPI app
app = FastAPI(
    title="Talk2Me API",
    description="Speech-to-Text and Text-to-Speech API with real-time conversation support",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure based on your needs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Initialize engines on startup."""
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
async def text_to_speech(request: dict[str, Any]) -> StreamingResponse:
    """Convert text to speech audio.

    Args:
        request: JSON with 'text' and optional 'voice' fields

    Returns:
        Audio stream (PCM 16-bit mono)
    """
    if not tts_engine:
        raise HTTPException(status_code=500, detail="TTS engine not initialized")

    try:
        text = request.get("text", "")
        if not text:
            raise HTTPException(status_code=400, detail="Text field is required")

        voice = request.get("voice")

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

                    except Exception as e:
                        logger.error(f"Processing failed: {e}")
                        await websocket.send_json({"type": "error", "message": str(e)})

                    # Reset wake word detector for next interaction
                    wake_word_detector.stop_listening()
                    wake_word_detector.start_listening()

            except WebSocketDisconnect:
                break

    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        # Cleanup
        wake_word_detector.stop_listening()
        logger.info("WebSocket connection closed")


if __name__ == "__main__":
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
