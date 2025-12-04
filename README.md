# Talk2Me

A fully offline, self-contained voice interaction system featuring speech-to-text, text-to-speech with voice cloning, and configurable wake word detection. Designed to run as a standalone service with comprehensive API endpoints.

## Features

### Speech-to-Text (STT)
- **Engine**: Vosk
- **Models**: Dual-model system
  - High-accuracy model for transcription
  - Lightweight model for fast wake word detection
- **Fully offline**: No internet connection required

### Text-to-Speech (TTS)
- **Engine**: XTTS v2 (Coqui TTS)
- **Voice Cloning**: Create and manage custom voice profiles
- **Multi-voice Support**: Switch between cloned voices on demand
- **Fully offline**: No cloud dependencies

### Wake Word System
- Configurable wake words to activate listening
- Configurable "start listening" phrases
- Configurable "done talking" phrases
- Low-latency detection using lightweight Vosk model

### Voice Cloning Management
- Persistent storage of cloned voice profiles
- Add/remove training samples at any time
- Automatic retraining only when samples change
- No manual retraining required for voice usage

### API Service
- RESTful API endpoints for all features
- WebSocket support for real-time streaming
- Designed for integration with other applications

## System Requirements

- Python 3.9+
- 4GB+ RAM (8GB recommended for voice cloning)
- ~5GB disk space for models and dependencies
- Microphone for speech input
- Speakers/audio output for TTS playback

## Project Structure

```
talk2me/
├── README.md
├── requirements.txt
├── setup.py
├── config/
│   ├── default.yaml          # Default configuration
│   └── voices.yaml           # Voice profiles configuration
├── models/
│   ├── vosk/
│   │   ├── vosk-model-en-us-0.22/      # High-accuracy STT model
│   │   └── vosk-model-small-en-us-0.15/ # Fast wake word model
│   └── xtts/
│       └── v2/               # XTTS v2 model files
├── voices/
│   └── [voice_name]/
│       ├── samples/          # Audio samples for cloning
│       └── metadata.json     # Voice profile metadata
├── src/
│   ├── __init__.py
│   ├── main.py               # Application entry point
│   ├── stt/
│   │   ├── __init__.py
│   │   ├── vosk_engine.py    # Vosk STT implementation
│   │   └── wake_word.py      # Wake word detection
│   ├── tts/
│   │   ├── __init__.py
│   │   ├── xtts_engine.py    # XTTS v2 implementation
│   │   └── voice_manager.py  # Voice cloning management
│   ├── api/
│   │   ├── __init__.py
│   │   ├── server.py         # FastAPI server
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── stt.py        # STT endpoints
│   │   │   ├── tts.py        # TTS endpoints
│   │   │   ├── voices.py     # Voice management endpoints
│   │   │   ├── config.py     # Configuration endpoints
│   │   │   └── websocket.py  # WebSocket handlers
│   │   └── models.py         # Pydantic models
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py         # Configuration management
│   │   ├── audio.py          # Audio utilities
│   │   └── events.py         # Event system
│   └── utils/
│       ├── __init__.py
│       ├── download.py       # Model downloader
│       └── logger.py         # Logging utilities
├── scripts/
│   ├── setup.sh              # Linux/macOS setup
│   ├── setup.bat             # Windows setup
│   └── download_models.py    # Model download script
├── tests/
│   ├── __init__.py
│   ├── test_stt.py
│   ├── test_tts.py
│   └── test_api.py
└── docker/
    ├── Dockerfile
    └── docker-compose.yml
```

## Installation

### Automated Setup

**Linux/macOS:**
```bash
git clone <repository-url>
cd talk2me
chmod +x scripts/setup.sh
./scripts/setup.sh
```

**Windows:**
```batch
git clone <repository-url>
cd talk2me
scripts\setup.bat
```

### Manual Setup

1. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   # or
   venv\Scripts\activate     # Windows
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download models:**
   ```bash
   python scripts/download_models.py
   ```

## Configuration

### Default Configuration (`config/default.yaml`)

```yaml
stt:
  model_path: "models/vosk/vosk-model-en-us-0.22"
  wake_word_model_path: "models/vosk/vosk-model-small-en-us-0.15"
  sample_rate: 16000

tts:
  model_path: "models/xtts/v2"
  default_voice: "default"
  sample_rate: 24000

wake_words:
  activation:
    - "hey talk to me"
    - "hello computer"
  start_listening:
    - "start listening"
    - "listen up"
  done_talking:
    - "done talking"
    - "that's all"
    - "stop listening"

api:
  host: "0.0.0.0"
  port: 8000
  cors_origins:
    - "*"

audio:
  input_device: null   # null = system default
  output_device: null  # null = system default
  chunk_size: 1024
```

### Voice Profiles (`config/voices.yaml`)

```yaml
voices:
  default:
    name: "Default Voice"
    samples_dir: "voices/default/samples"
    language: "en"

  custom_voice_1:
    name: "My Custom Voice"
    samples_dir: "voices/custom_voice_1/samples"
    language: "en"
```

## Usage

### Starting the Service

```bash
# Start with default configuration
python -m src.main

# Start with custom config
python -m src.main --config path/to/config.yaml

# Start API server only
python -m src.main --api-only

# Start with specific port
python -m src.main --port 9000
```

### Interactive Mode

```bash
python -m src.main --interactive
```

## API Reference

### Base URL
```
http://localhost:8000/api/v1
```

### Speech-to-Text Endpoints

#### `POST /stt/transcribe`
Transcribe audio file to text.

**Request:**
- Content-Type: `multipart/form-data`
- Body: `audio` (file)

**Response:**
```json
{
  "text": "transcribed text here",
  "confidence": 0.95,
  "duration": 2.5
}
```

#### `WebSocket /stt/stream`
Real-time streaming transcription.

**Messages:**
- Send: Binary audio chunks
- Receive: JSON with partial/final transcriptions

---

### Text-to-Speech Endpoints

#### `POST /tts/synthesize`
Convert text to speech.

**Request:**
```json
{
  "text": "Hello, world!",
  "voice": "default",
  "language": "en"
}
```

**Response:**
- Content-Type: `audio/wav`
- Body: Audio binary data

#### `POST /tts/synthesize/stream`
Stream synthesized audio.

**Request:**
```json
{
  "text": "Long text to synthesize...",
  "voice": "default"
}
```

**Response:**
- Content-Type: `audio/wav`
- Transfer-Encoding: chunked

---

### Voice Management Endpoints

#### `GET /voices`
List all available voices.

**Response:**
```json
{
  "voices": [
    {
      "id": "default",
      "name": "Default Voice",
      "samples_count": 3,
      "language": "en"
    }
  ]
}
```

#### `POST /voices`
Create a new voice profile.

**Request:**
```json
{
  "id": "new_voice",
  "name": "My New Voice",
  "language": "en"
}
```

#### `GET /voices/{voice_id}`
Get voice profile details.

#### `DELETE /voices/{voice_id}`
Delete a voice profile.

#### `POST /voices/{voice_id}/samples`
Add sample audio to voice profile.

**Request:**
- Content-Type: `multipart/form-data`
- Body: `audio` (file)

#### `GET /voices/{voice_id}/samples`
List all samples for a voice.

#### `DELETE /voices/{voice_id}/samples/{sample_id}`
Remove a sample from voice profile.

---

### Configuration Endpoints

#### `GET /config`
Get current configuration.

#### `PATCH /config`
Update configuration.

**Request:**
```json
{
  "wake_words": {
    "activation": ["hey computer", "wake up"]
  }
}
```

#### `GET /config/wake-words`
Get wake word configuration.

#### `PUT /config/wake-words`
Update wake word configuration.

---

### System Endpoints

#### `GET /health`
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "stt_loaded": true,
  "tts_loaded": true,
  "version": "1.0.0"
}
```

#### `GET /status`
Detailed system status.

#### `POST /reload`
Reload models and configuration.

---

### WebSocket Endpoints

#### `WebSocket /ws/conversation`
Full-duplex conversation mode.

**Client -> Server Messages:**
```json
{"type": "audio", "data": "<base64 audio>"}
{"type": "config", "wake_words": ["hey"]}
{"type": "command", "action": "start_listening"}
```

**Server -> Client Messages:**
```json
{"type": "transcription", "text": "hello", "final": true}
{"type": "audio", "data": "<base64 audio>"}
{"type": "status", "listening": true}
{"type": "wake_word_detected", "word": "hey talk to me"}
```

## Voice Cloning

### Adding a New Voice

1. **Create voice profile:**
   ```bash
   curl -X POST http://localhost:8000/api/v1/voices \
     -H "Content-Type: application/json" \
     -d '{"id": "my_voice", "name": "My Voice", "language": "en"}'
   ```

2. **Add audio samples:**
   ```bash
   curl -X POST http://localhost:8000/api/v1/voices/my_voice/samples \
     -F "audio=@sample1.wav"
   ```

3. **Use the voice:**
   ```bash
   curl -X POST http://localhost:8000/api/v1/tts/synthesize \
     -H "Content-Type: application/json" \
     -d '{"text": "Hello!", "voice": "my_voice"}' \
     --output output.wav
   ```

### Sample Requirements

- Format: WAV, MP3, FLAC, or OGG
- Duration: 6-30 seconds per sample (10-15 seconds optimal)
- Quality: Clear audio, minimal background noise
- Content: Natural speech, varied intonation
- Quantity: 1-10 samples (3-5 recommended)

### Managing Samples

Samples can be added or removed at any time. The voice will automatically use the updated samples on the next synthesis request without requiring manual retraining.

## Offline Operation

Talk2Me is designed for complete offline operation:

1. **All models bundled**: STT and TTS models included in distribution
2. **No API calls**: No external service dependencies
3. **Local processing**: All computation runs locally
4. **Portable**: Single folder contains everything needed

### First-Time Setup

On first run, the setup script downloads required models (~4GB). After setup, no internet connection is needed.

## Building for Distribution

### Create Standalone Package

```bash
python scripts/build.py --target linux   # Linux bundle
python scripts/build.py --target windows # Windows bundle
python scripts/build.py --target macos   # macOS bundle
```

### Docker

```bash
cd docker
docker-compose up -d
```

## Development

### Running Tests

```bash
pytest tests/
```

### Code Style

```bash
black src/
isort src/
flake8 src/
```

## Troubleshooting

### Common Issues

**"No audio input device found"**
- Ensure microphone is connected
- Check system audio settings
- Set specific device in config

**"Model not found"**
- Run `python scripts/download_models.py`
- Check `models/` directory structure

**"CUDA out of memory"**
- Reduce batch size in config
- Use CPU mode: `--device cpu`

**"Voice cloning quality poor"**
- Add more diverse samples
- Ensure samples are clear audio
- Use 10-15 second clips

## License

[License Type] - See LICENSE file for details.

## Acknowledgments

- [Vosk](https://alphacephei.com/vosk/) - Offline speech recognition
- [Coqui TTS](https://github.com/coqui-ai/TTS) - XTTS v2 text-to-speech
- [FastAPI](https://fastapi.tiangolo.com/) - API framework
