# Talk2Me API Documentation

## Overview

Talk2Me is a fully offline, self-contained voice interaction system featuring speech-to-text (STT), text-to-speech (TTS) with voice cloning, and configurable wake word detection. This documentation provides comprehensive guidance for AI agents to understand and implement Talk2Me in new projects.

## Table of Contents

1. [System Architecture](#system-architecture)
2. [REST API Reference](#rest-api-reference)
3. [WebSocket API](#websocket-api)
4. [Configuration Files](#configuration-files)
5. [CLI Interface](#cli-interface)
6. [Integration Guides](#integration-guides)
7. [Code Examples](#code-examples)
8. [Error Handling](#error-handling)
9. [Model Requirements](#model-requirements)
10. [Setup and Deployment](#setup-and-deployment)

## System Architecture

Talk2Me consists of four main components:

- **STT Engine**: Uses Vosk for offline speech recognition
- **TTS Engine**: Uses XTTS v2 (Coqui TTS) for multilingual speech synthesis with voice cloning
- **Wake Word Detector**: Lightweight Vosk model for real-time wake word detection
- **API Server**: FastAPI-based REST and WebSocket endpoints

### Data Flow

```
Audio Input → Wake Word Detection → STT Transcription → Response Generation → TTS Synthesis → Audio Output
```

## REST API Reference

Base URL: `http://localhost:8000`

### Core Endpoints

#### GET /

Root endpoint for health checking.

**Response:**
```json
{
  "message": "Talk2Me API is running"
}
```

#### POST /stt

Convert speech audio to text.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: `file` (UploadFile) - Audio file (WAV, PCM 16-bit mono)
- Query Parameters:
  - `sample_rate` (optional): int - Sample rate override (default: 16000)

**Response:**
```json
{
  "text": "transcribed text content"
}
```

**Status Codes:**
- `200`: Success
- `500`: STT engine not initialized or transcription failed

#### POST /tts

Convert text to speech audio.

**Request:**
- Method: `POST`
- Content-Type: `application/json`
- Body:
```json
{
  "text": "Text to synthesize",
  "voice": "voice_name"
}
```

**Response:**
- Content-Type: `audio/wav`
- Body: PCM 16-bit mono audio data

**Status Codes:**
- `200`: Success
- `400`: Invalid text or voice
- `500`: TTS engine not initialized or synthesis failed

### Voice Management Endpoints

#### GET /voices

List all available voices.

**Response:**
```json
{
  "voices": [
    {
      "name": "Default Voice",
      "display_name": "Default Voice",
      "language": "en"
    }
  ]
}
```

#### POST /voices

Create a new voice profile.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Form Data:
  - `name`: str - Display name for the voice
  - `language`: str - Language code (default: "en")
  - `samples`: list[UploadFile] - Audio sample files (WAV format)

**Response:**
```json
{
  "voice_id": "generated_voice_id",
  "message": "Voice created successfully",
  "samples_uploaded": 3
}
```

#### PUT /voices/{voice_id}

Update voice profile.

**Request:**
- Method: `PUT`
- Path Parameters: `voice_id` (str)
- Content-Type: `application/json`
- Body:
```json
{
  "name": "New Display Name",
  "language": "es"
}
```

**Response:**
```json
{
  "message": "Voice updated successfully"
}
```

#### DELETE /voices/{voice_id}

Delete voice profile and samples.

**Request:**
- Method: `DELETE`
- Path Parameters: `voice_id` (str)

**Response:**
```json
{
  "message": "Voice deleted successfully"
}
```

#### POST /voices/{voice_id}/samples

Upload audio samples for voice.

**Request:**
- Method: `POST`
- Path Parameters: `voice_id` (str)
- Content-Type: `multipart/form-data`
- Body: `samples` (list[UploadFile]) - WAV audio files

**Response:**
```json
{
  "uploaded": ["sample1.wav", "sample2.wav"],
  "message": "Successfully uploaded 2 samples"
}
```

#### DELETE /voices/{voice_id}/samples/{sample_id}

Remove specific sample from voice.

**Request:**
- Method: `DELETE`
- Path Parameters:
  - `voice_id` (str)
  - `sample_id` (str) - Filename of sample to remove

**Response:**
```json
{
  "message": "Sample deleted successfully"
}
```

#### POST /voices/{voice_id}/retrain

Trigger voice retraining (validates samples).

**Request:**
- Method: `POST`
- Path Parameters: `voice_id` (str)

**Response:**
```json
{
  "message": "Voice retrained successfully",
  "sample_count": 5
}
```

## WebSocket API

WebSocket endpoint for real-time conversation: `ws://localhost:8000/ws`

### Connection Protocol

1. Client connects to `/ws`
2. Server accepts connection and starts wake word detection
3. Client sends audio chunks as binary data
4. Server processes audio for wake word detection
5. When wake word detected, server transcribes speech and responds

### Message Types

#### Client → Server

**Audio Data:**
- Type: Binary
- Data: PCM 16-bit mono audio chunks

#### Server → Client

**Transcription:**
```json
{
  "type": "transcription",
  "text": "Hello, how can I help you?"
}
```

**Error:**
```json
{
  "type": "error",
  "message": "Processing failed: audio format invalid"
}
```

### Connection States

- **Listening**: Wake word detector active, waiting for activation phrase
- **Processing**: Wake word detected, transcribing speech
- **Responding**: Generating TTS response

### Error Handling

- `1011`: Internal server error (engines not initialized)
- Connection closes automatically after successful interaction
- Reconnect required for new conversation

## Configuration Files

### config/default.yaml

Main configuration file with system settings:

```yaml
stt:
  model_path: "models/vosk-model-small-en-us-0.15"
  wake_word_model_path: "models/vosk-model-small-en-us-0.15"
  sample_rate: 16000

tts:
  model_path: "models/models/xtts/v2"
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
  input_device: null  # null = system default
  output_device: null # null = system default
  chunk_size: 1024
```

### config/voices.yaml

Voice profiles configuration:

```yaml
voices:
  default:
    name: "Default Voice"
    samples_dir: "voices/default/samples"
    language: "en"

  custom_voice:
    name: "My Custom Voice"
    samples_dir: "voices/custom_voice/samples"
    language: "en"
```

### Configuration Schema

#### STT Configuration
- `model_path`: Path to Vosk STT model directory
- `wake_word_model_path`: Path to lightweight Vosk model for wake word detection
- `sample_rate`: Audio sample rate (default: 16000)

#### TTS Configuration
- `model_path`: Path to XTTS v2 model directory
- `default_voice`: Default voice identifier
- `sample_rate`: Output audio sample rate (default: 24000)

#### Wake Words Configuration
- `activation`: List of phrases to activate listening
- `start_listening`: Alternative activation phrases
- `done_talking`: Phrases to stop listening

#### API Configuration
- `host`: Server bind address
- `port`: Server port number
- `cors_origins`: Allowed CORS origins

#### Audio Configuration
- `input_device`: Audio input device index (null for default)
- `output_device`: Audio output device index (null for default)
- `chunk_size`: Audio processing chunk size

## CLI Interface

Command-line interface for running Talk2Me in different modes.

### Basic Usage

```bash
# Start API server with default config
talk2me

# Start with custom configuration
talk2me --config path/to/config.yaml

# Start interactive voice mode
talk2me --interactive

# Start API server on specific port
talk2me --port 9000
```

### Command Line Options

- `--config`: Path to configuration file (default: config/default.yaml)
- `--interactive`: Start interactive voice conversation mode
- `--api-only`: Run API server only (default mode)
- `--host`: API server host (default: 0.0.0.0)
- `--port`: API server port (default: 8000)

### Interactive Mode

Requires PyAudio for microphone access:

```bash
pip install pyaudio
talk2me --interactive
```

Features:
- Real-time wake word detection
- Automatic speech transcription
- Echo responses via TTS
- Press Ctrl+C to exit

### API Server Mode

Default mode providing REST and WebSocket endpoints:

```bash
talk2me --api-only --host 127.0.0.1 --port 8000
```

## Integration Guides

### Basic STT/TTS Integration

#### Python Example

```python
import requests

# STT: Upload audio file
with open('audio.wav', 'rb') as f:
    response = requests.post('http://localhost:8000/stt', files={'file': f})
    text = response.json()['text']

# TTS: Generate speech
response = requests.post('http://localhost:8000/tts',
                        json={'text': 'Hello world', 'voice': 'default'})
with open('output.wav', 'wb') as f:
    f.write(response.content)
```

### Voice Cloning Integration

#### Create Custom Voice

```python
import requests

# Create voice profile
response = requests.post('http://localhost:8000/voices',
                        data={'name': 'My Voice', 'language': 'en'})
voice_id = response.json()['voice_id']

# Upload samples
files = [('samples', open(f'sample{i}.wav', 'rb')) for i in range(3)]
response = requests.post(f'http://localhost:8000/voices/{voice_id}/samples',
                        files=files)
```

#### Use Custom Voice

```python
response = requests.post('http://localhost:8000/tts',
                        json={'text': 'Hello', 'voice': voice_id})
```

### Wake Word Detection Integration

#### WebSocket Real-time Conversation

```python
import websocket
import json

def on_message(ws, message):
    data = json.loads(message)
    if data['type'] == 'transcription':
        print(f"Transcribed: {data['text']}")

ws = websocket.WebSocketApp("ws://localhost:8000/ws",
                           on_message=on_message)
ws.run_forever()
```

### Interactive Mode Integration

For applications requiring full voice interaction:

```python
from talk2me.cli import InteractiveMode

# Initialize with custom config
interactive = InteractiveMode(config_path='config/custom.yaml')
interactive.run()
```

## Code Examples

### Python SDK Usage

#### Basic STT

```python
from talk2me.stt.engine import STTEngine

# Initialize engine
stt = STTEngine()

# Transcribe audio
with open('audio.wav', 'rb') as f:
    audio_data = f.read()

text = stt.transcribe(audio_data)
print(f"Transcribed: {text}")
```

#### Basic TTS

```python
from talk2me.tts.engine import TTSEngine

# Initialize engine
tts = TTSEngine()

# Synthesize speech
audio_bytes = tts.synthesize("Hello world", voice_name="default")

# Save to file
with open('output.wav', 'wb') as f:
    f.write(audio_bytes)
```

#### Voice Cloning

```python
from talk2me.tts.engine import TTSEngine
import os

# Initialize engine
tts = TTSEngine()

# Create voice directory
voice_dir = "voices/my_voice/samples"
os.makedirs(voice_dir, exist_ok=True)

# Add voice to config
tts.voices['my_voice'] = {
    'name': 'My Voice',
    'samples_dir': voice_dir,
    'language': 'en'
}

# Synthesize with custom voice
audio = tts.synthesize("Hello from my voice", voice_name="my_voice")
```

#### Wake Word Detection

```python
from talk2me.core.wake_word import WakeWordDetector

# Initialize detector
detector = WakeWordDetector()

# Start listening
detector.start_listening()

# Process audio chunks
while True:
    audio_chunk = get_audio_chunk()  # Your audio capture function
    detector.add_audio_chunk(audio_chunk)

    if detector.is_wake_word_detected():
        print("Wake word detected!")
        break

detector.stop_listening()
```

### FastAPI Integration

#### Custom Endpoint

```python
from fastapi import FastAPI
from talk2me.stt.engine import STTEngine
from talk2me.tts.engine import TTSEngine

app = FastAPI()
stt = STTEngine()
tts = TTSEngine()

@app.post("/process_audio")
async def process_audio(file: UploadFile):
    # Read audio
    audio_data = await file.read()

    # Transcribe
    text = stt.transcribe(audio_data)

    # Generate response
    response_text = f"You said: {text}"

    # Synthesize
    audio_response = tts.synthesize(response_text)

    return StreamingResponse(
        iter([audio_response]),
        media_type="audio/wav"
    )
```

### WebSocket Client

#### JavaScript Example

```javascript
const ws = new WebSocket('ws://localhost:8000/ws');

ws.onopen = () => {
    console.log('Connected to Talk2Me');
};

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.type === 'transcription') {
        console.log('Transcribed:', data.text);
    }
};

// Send audio chunks
function sendAudioChunk(audioData) {
    ws.send(audioData);
}
```

## Error Handling

### HTTP Status Codes

- `200`: Success
- `400`: Bad Request (invalid parameters)
- `404`: Not Found (voice or resource not found)
- `500`: Internal Server Error (engine failure)

### Common Error Responses

#### STT Errors

```json
{
  "detail": "Transcription failed: audio format invalid"
}
```

#### TTS Errors

```json
{
  "detail": "Voice 'unknown' not configured"
}
```

#### Voice Management Errors

```json
{
  "detail": "Voice already exists"
}
```

### WebSocket Error Handling

```javascript
ws.onerror = (error) => {
    console.error('WebSocket error:', error);
};

ws.onclose = (event) => {
    if (event.code === 1011) {
        console.error('Server error: engines not initialized');
    }
};
```

### Exception Types

- `ValueError`: Invalid configuration or parameters
- `FileNotFoundError`: Missing model or config files
- `RuntimeError`: Engine initialization or processing failures
- `HTTPException`: API-specific errors

## Model Requirements

### STT Models

- **Vosk Model**: `vosk-model-small-en-us-0.15` (~40MB)
- **Location**: `models/vosk-model-small-en-us-0.15/`
- **Purpose**: High-accuracy speech transcription

### Wake Word Models

- **Vosk Model**: `vosk-model-small-en-us-0.15` (same as STT)
- **Purpose**: Lightweight wake word detection

### TTS Models

- **XTTS v2 Model**: `models/models/xtts/v2/` (~4GB)
- **Engine**: Coqui TTS XTTS v2
- **Features**: Multilingual, voice cloning capable

### Model Download

Models are downloaded automatically via setup script:

```bash
python scripts/download_models.py
```

### Disk Space Requirements

- STT Model: ~40MB
- TTS Model: ~4GB
- Voice Samples: Variable (recommended: 100MB per voice)
- Total: ~4.5GB minimum

## Setup and Deployment

### Prerequisites

- Python 3.9+
- 8GB+ RAM (16GB recommended)
- Microphone and speakers (for interactive mode)
- ~5GB disk space

### Installation

#### Automated Setup

```bash
# Linux/macOS
chmod +x scripts/setup.sh
./scripts/setup.sh

# Windows
scripts\setup.bat
```

#### Manual Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Download models
python scripts/download_models.py
```

### Configuration

1. **Edit config/default.yaml** for system settings
2. **Edit config/voices.yaml** for voice profiles
3. **Create voice directories** under `voices/`

### Running the Application

#### API Server

```bash
# Activate virtual environment
source venv/bin/activate

# Start server
talk2me

# Or with custom config
talk2me --config config/production.yaml --port 8000
```

#### Interactive Mode

```bash
talk2me --interactive
```

### Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . .

RUN pip install -r requirements.txt
RUN python scripts/download_models.py

EXPOSE 8000
CMD ["python", "-m", "talk2me.api.main"]
```

### Production Deployment

#### Systemd Service

```ini
[Unit]
Description=Talk2Me API Server
After=network.target

[Service]
Type=simple
User=talk2me
WorkingDirectory=/opt/talk2me
ExecStart=/opt/talk2me/venv/bin/talk2me --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

#### Nginx Reverse Proxy

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    location /ws {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

### Monitoring and Logging

- Logs are written to stdout/stderr
- Configure log levels in Python logging
- Monitor API endpoints with health checks
- Track voice cloning performance

### Troubleshooting

#### Common Issues

1. **"Model not found"**
   - Run `python scripts/download_models.py`
   - Check `models/` directory permissions

2. **"Audio device not found"**
   - Verify microphone/speaker connections
   - Check audio device permissions

3. **"CUDA out of memory"**
   - Reduce TTS batch size
   - Use CPU mode if available

4. **"Voice cloning quality poor"**
   - Add more diverse audio samples
   - Ensure samples are clean recordings

### Performance Optimization

- Use SSD storage for models
- Pre-load models on startup
- Configure appropriate audio buffer sizes
- Monitor memory usage with large voice libraries

---

This documentation provides complete guidance for implementing Talk2Me in AI agent workflows. All endpoints, configurations, and integration patterns are designed for programmatic automation and reliable operation.