from io import BytesIO
from unittest.mock import Mock, mock_open, patch

import pytest
from fastapi import WebSocketDisconnect
from fastapi.testclient import TestClient

from talk2me.api.main import app


@pytest.fixture
def client():
    """Test client fixture."""
    return TestClient(app)


@pytest.fixture
def mock_engines():
    """Mock engines fixture."""
    mock_stt = Mock()
    mock_tts = Mock()
    mock_wake = Mock()

    # Mock voices for TTS
    mock_tts.voices = {
        "voice1": {"name": "Voice One", "language": "en"},
        "voice2": {"name": "Voice Two", "language": "es"},
    }

    return mock_stt, mock_tts, mock_wake


class TestAPI:
    """Test suite for API endpoints."""

    def test_root_endpoint(self, client):
        """Test root endpoint returns correct message."""
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {"message": "Talk2Me API is running"}

    @patch("talk2me.api.main.stt_engine", None)
    def test_stt_engine_not_initialized(self, client):
        """Test STT endpoint when engine is not initialized."""
        response = client.post("/stt", files={"file": ("test.wav", b"audio_data")})
        assert response.status_code == 500
        assert "STT engine not initialized" in response.json()["detail"]

    @patch("talk2me.api.main.stt_engine")
    def test_stt_success(self, mock_stt, client):
        """Test successful STT transcription."""
        mock_stt.transcribe.return_value = "Hello world"

        response = client.post("/stt", files={"file": ("test.wav", b"audio_data")})

        assert response.status_code == 200
        assert response.json() == {"text": "Hello world"}
        mock_stt.transcribe.assert_called_once_with(b"audio_data", None)

    @patch("talk2me.api.main.stt_engine")
    def test_stt_with_sample_rate(self, mock_stt, client):
        """Test STT with custom sample rate."""
        mock_stt.transcribe.return_value = "Test transcription"

        response = client.post(
            "/stt?sample_rate=22050", files={"file": ("test.wav", b"audio_data")}
        )

        assert response.status_code == 200
        mock_stt.transcribe.assert_called_once_with(b"audio_data", 22050)

    @patch("talk2me.api.main.stt_engine")
    def test_stt_transcription_error(self, mock_stt, client):
        """Test STT when transcription fails."""
        mock_stt.transcribe.side_effect = Exception("Transcription failed")

        response = client.post("/stt", files={"file": ("test.wav", b"audio_data")})

        assert response.status_code == 500
        assert "Transcription failed" in response.json()["detail"]

    @patch("talk2me.api.main.tts_engine", None)
    def test_tts_engine_not_initialized(self, client):
        """Test TTS endpoint when engine is not initialized."""
        response = client.post("/tts", json={"text": "Hello"})
        assert response.status_code == 500
        assert "TTS engine not initialized" in response.json()["detail"]

    @patch("talk2me.api.main.tts_engine")
    def test_tts_success(self, mock_tts, client):
        """Test successful TTS synthesis."""
        mock_tts.synthesize.return_value = b"audio_bytes"

        response = client.post("/tts", json={"text": "Hello world"})

        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/wav"
        assert response.content == b"audio_bytes"
        mock_tts.synthesize.assert_called_once_with("Hello world", None)

    @patch("talk2me.api.main.tts_engine")
    def test_tts_with_voice(self, mock_tts, client):
        """Test TTS with specific voice."""
        mock_tts.synthesize.return_value = b"audio_bytes"

        response = client.post("/tts", json={"text": "Hello", "voice": "voice1"})

        assert response.status_code == 200
        mock_tts.synthesize.assert_called_once_with("Hello", "voice1")

    @patch("talk2me.api.main.tts_engine")
    def test_tts_missing_text(self, _mock_tts, client):
        """Test TTS with missing text field."""
        response = client.post("/tts", json={})

        assert response.status_code == 422
        assert response.json()["detail"][0]["msg"] == "Field required"

    @patch("talk2me.api.main.tts_engine")
    def test_tts_empty_text(self, _mock_tts, client):
        """Test TTS with empty text."""
        response = client.post("/tts", json={"text": ""})

        assert response.status_code == 422
        assert response.json()["detail"][0]["msg"] == "String should have at least 1 character"

    @patch("talk2me.api.main.tts_engine")
    def test_tts_synthesis_error(self, mock_tts, client):
        """Test TTS when synthesis fails."""
        mock_tts.synthesize.side_effect = ValueError("Invalid voice")

        response = client.post("/tts", json={"text": "Hello", "voice": "invalid"})

        assert response.status_code == 400
        assert "Invalid voice" in response.json()["detail"]

    @patch("talk2me.api.main.tts_engine")
    def test_tts_runtime_error(self, mock_tts, client):
        """Test TTS when synthesis raises runtime error."""
        mock_tts.synthesize.side_effect = RuntimeError("Synthesis failed")

        response = client.post("/tts", json={"text": "Hello"})

        assert response.status_code == 500
        assert "Synthesis failed" in response.json()["detail"]

    @patch("talk2me.api.main.tts_engine", None)
    def test_voices_engine_not_initialized(self, client):
        """Test voices endpoint when engine is not initialized."""
        response = client.get("/voices")
        assert response.status_code == 500
        assert "TTS engine not initialized" in response.json()["detail"]

    @patch("talk2me.api.main.tts_engine")
    def test_voices_success(self, mock_tts, client):
        """Test successful voices listing."""
        mock_tts.voices = {
            "voice1": {"name": "Voice One", "language": "en"},
            "voice2": {"name": "Voice Two", "language": "es"},
        }

        response = client.get("/voices")

        assert response.status_code == 200
        voices = response.json()["voices"]
        assert len(voices) == 2
        assert voices[0]["name"] == "voice1"
        assert voices[0]["display_name"] == "Voice One"
        assert voices[0]["language"] == "en"

    @patch("talk2me.api.main.tts_engine")
    def test_voices_error(self, mock_tts, client):
        """Test voices endpoint when listing fails."""
        mock_tts.voices = {}
        # Simulate error by making voices access fail
        mock_tts.voices = Mock(side_effect=Exception("Database error"))

        response = client.get("/voices")

        assert response.status_code == 500
        assert "Failed to list voices" in response.json()["detail"]

    @patch("talk2me.api.main.stt_engine", None)
    @patch("talk2me.api.main.tts_engine", None)
    @patch("talk2me.api.main.wake_word_detector", None)
    def test_websocket_engines_not_initialized(self, client):
        """Test WebSocket connection when engines are not initialized."""
        with pytest.raises(WebSocketDisconnect), client.websocket_connect("/ws"):
            pass

    @patch("talk2me.api.main.stt_engine")
    @patch("talk2me.api.main.tts_engine")
    @patch("talk2me.api.main.wake_word_detector")
    def test_websocket_connection_success(self, _mock_wake, _mock_tts, _mock_stt, client):
        """Test successful WebSocket connection."""
        with client.websocket_connect("/ws"):
            # Should connect successfully
            pass

    @patch("talk2me.api.main.stt_engine")
    @patch("talk2me.api.main.tts_engine")
    @patch("talk2me.api.main.wake_word_detector")
    def test_websocket_audio_processing(self, mock_wake, mock_tts, mock_stt, client):
        """Test WebSocket audio processing and wake word detection."""
        mock_wake.is_wake_word_detected.return_value = True
        mock_stt.transcribe.return_value = "Hello"
        mock_tts.synthesize.return_value = b"audio_response"

        with client.websocket_connect("/ws") as websocket:
            # Send audio data
            websocket.send_bytes(b"audio_chunk")

            # Should receive transcription
            response = websocket.receive_json()
            assert response["type"] == "transcription"
            assert response["text"] == "Hello"

            # Should receive audio response
            audio_response = websocket.receive_bytes()
            assert audio_response == b"audio_response"

    @patch("talk2me.api.main.stt_engine")
    @patch("talk2me.api.main.tts_engine")
    @patch("talk2me.api.main.wake_word_detector")
    def test_websocket_no_wake_word(self, mock_wake, _mock_tts, _mock_stt, client):
        """Test WebSocket when wake word is not detected."""
        mock_wake.is_wake_word_detected.return_value = False

        with client.websocket_connect("/ws") as websocket:
            websocket.send_bytes(b"audio_chunk")
            # Should not receive any response, connection closed by handler

    @patch("talk2me.api.main.stt_engine")
    @patch("talk2me.api.main.tts_engine")
    @patch("talk2me.api.main.wake_word_detector")
    def test_websocket_empty_transcription(self, mock_wake, _mock_tts, _mock_stt, client):
        """Test WebSocket when transcription is empty."""
        mock_wake.is_wake_word_detected.return_value = True
        _mock_stt.transcribe.return_value = ""  # Empty transcription

        with client.websocket_connect("/ws") as websocket:
            websocket.send_bytes(b"audio_chunk")
            # Should not receive transcription or audio, connection closed by handler

    @patch("talk2me.api.main.stt_engine")
    @patch("talk2me.api.main.tts_engine")
    @patch("talk2me.api.main.wake_word_detector")
    def test_websocket_transcription_error(self, mock_wake, _mock_tts, _mock_stt, client):
        """Test WebSocket when transcription fails."""
        mock_wake.is_wake_word_detected.return_value = True
        _mock_stt.transcribe.side_effect = Exception("Transcription error")

        with client.websocket_connect("/ws") as websocket:
            websocket.send_bytes(b"audio_chunk")

            # Should receive error message
            response = websocket.receive_json()
            assert response["type"] == "error"
            assert "Transcription error" in response["message"]

    @patch("talk2me.api.main.stt_engine")
    @patch("talk2me.api.main.tts_engine")
    @patch("talk2me.api.main.wake_word_detector")
    def test_websocket_synthesis_error(self, mock_wake, mock_tts, mock_stt, client):
        """Test WebSocket when synthesis fails."""
        mock_wake.is_wake_word_detected.return_value = True
        mock_stt.transcribe.return_value = "Hello"
        mock_tts.synthesize.side_effect = Exception("Synthesis error")

        with client.websocket_connect("/ws") as websocket:
            websocket.send_bytes(b"audio_chunk")

            # Should receive transcription
            response = websocket.receive_json()
            assert response["type"] == "transcription"
            assert response["text"] == "Hello"

            # Should receive error on synthesis
            error_response = websocket.receive_json()
            assert error_response["type"] == "error"
            assert "Synthesis error" in error_response["message"]

    @patch("talk2me.api.main.stt_engine")
    @patch("talk2me.api.main.tts_engine")
    @patch("talk2me.api.main.wake_word_detector")
    def test_websocket_disconnect(self, _mock_wake, _mock_tts, _mock_stt, client):
        """Test WebSocket disconnect handling."""
        with client.websocket_connect("/ws") as websocket:
            websocket.close()
            # Should handle disconnect gracefully

    @patch("talk2me.api.main.tts_engine", None)
    def test_create_voice_engine_not_initialized(self, client):
        """Test create voice when TTS engine not initialized."""
        response = client.post("/voices", params={"name": "Test Voice"})
        assert response.status_code == 500
        assert "TTS engine not initialized" in response.json()["detail"]

    @patch("talk2me.api.main.tts_engine")
    @patch("talk2me.api.main.os.makedirs")
    @patch("talk2me.api.main.yaml.dump")
    def test_create_voice_success(self, mock_yaml_dump, _mock_makedirs, mock_tts, client):
        """Test successful voice creation."""
        mock_tts.voices = {}
        mock_file = Mock()
        mock_file.filename = "sample.wav"

        with patch("builtins.open", mock_open()):
            response = client.post(
                "/voices",
                params={"name": "Test Voice", "language": "en"},
                files={"samples": ("sample.wav", BytesIO(b"audio_data"), "audio/wav")},
            )

        assert response.status_code == 200
        result = response.json()
        assert "voice_id" in result
        assert result["message"] == "Voice created successfully"
        assert result["samples_uploaded"] == 1
        mock_yaml_dump.assert_called_once()

    @patch("talk2me.api.main.tts_engine")
    def test_create_voice_missing_name(self, mock_tts, client):
        """Test create voice with missing name."""
        mock_tts.voices = {}
        response = client.post("/voices", data={})
        assert response.status_code == 422
        detail = response.json()["detail"]
        assert isinstance(detail, list)
        assert len(detail) == 1
        assert detail[0]["loc"] == ["query", "name"]
        assert detail[0]["msg"] == "Field required"
        assert detail[0]["type"] == "missing"

    @patch("talk2me.api.main.tts_engine")
    def test_create_voice_already_exists(self, mock_tts, client):
        """Test create voice that already exists."""
        mock_tts.voices = {"test_voice": {}}
        response = client.post("/voices", params={"name": "Test Voice"})
        assert response.status_code == 400
        assert "Voice already exists" in response.json()["detail"]

    @patch("talk2me.api.main.tts_engine")
    def test_create_voice_invalid_file_type(self, mock_tts, client):
        """Test create voice with invalid file type."""
        mock_tts.voices = {}
        response = client.post(
            "/voices",
            params={"name": "Test Voice"},
            files={"samples": ("sample.mp3", BytesIO(b"audio_data"), "audio/mp3")},
        )
        assert response.status_code == 400
        assert "Only WAV files are allowed" in response.json()["detail"]

    @patch("talk2me.api.main.tts_engine")
    @patch("talk2me.api.main.os.makedirs", side_effect=OSError("Permission denied"))
    def test_create_voice_makedirs_failure(self, _mock_makedirs, mock_tts, client):
        """Test create voice when directory creation fails."""
        mock_tts.voices = {}
        response = client.post("/voices", params={"name": "Test Voice"})
        assert response.status_code == 500
        # The exception is not caught, so FastAPI returns 500

    @patch("talk2me.api.main.tts_engine", None)
    def test_update_voice_engine_not_initialized(self, client):
        """Test update voice when TTS engine not initialized."""
        response = client.put("/voices/test_voice", data={"name": "New Name"})
        assert response.status_code == 500

    @patch("talk2me.api.main.tts_engine")
    @patch("talk2me.api.main.yaml.dump")
    @patch("talk2me.api.main.os.path.exists", return_value=False)
    def test_update_voice_success(self, _mock_exists, mock_yaml_dump, mock_tts, client):
        """Test successful voice update."""
        mock_tts.voices = {"test_voice": {"name": "Old Name", "language": "en"}}
        response = client.put("/voices/test_voice", params={"name": "New Name", "language": "es"})
        assert response.status_code == 200
        assert response.json()["message"] == "Voice updated successfully"
        assert mock_tts.voices["New Name"]["name"] == "New Name"
        assert mock_tts.voices["New Name"]["language"] == "es"
        assert "test_voice" not in mock_tts.voices
        mock_yaml_dump.assert_called_once()

    @patch("talk2me.api.main.tts_engine")
    def test_update_voice_not_found(self, mock_tts, client):
        """Test update voice that doesn't exist."""
        mock_tts.voices = {}
        response = client.put("/voices/nonexistent", data={"name": "New Name"})
        assert response.status_code == 404
        assert "Voice not found" in response.json()["detail"]

    @patch("talk2me.api.main.tts_engine")
    @patch("talk2me.api.main.yaml.dump", side_effect=Exception("YAML dump failed"))
    def test_update_voice_yaml_save_failure(self, _mock_yaml_dump, mock_tts, client):
        """Test update voice when saving config fails."""
        mock_tts.voices = {"test_voice": {"name": "Old Name", "language": "en"}}
        response = client.put("/voices/test_voice", params={"name": "New Name"})
        assert response.status_code == 500
        assert "Failed to save voice configuration" in response.json()["detail"]

    @patch("talk2me.api.main.tts_engine", None)
    def test_delete_voice_engine_not_initialized(self, client):
        """Test delete voice when TTS engine not initialized."""
        response = client.delete("/voices/test_voice")
        assert response.status_code == 500

    @patch("talk2me.api.main.tts_engine")
    @patch("talk2me.api.main.shutil.rmtree")
    @patch("talk2me.api.main.os.path.exists", return_value=True)
    @patch("talk2me.api.main.yaml.dump")
    def test_delete_voice_success(
        self, mock_yaml_dump, _mock_exists, mock_rmtree, mock_tts, client
    ):
        """Test successful voice deletion."""
        mock_tts.voices = {"test_voice": {"samples_dir": "voices/test_voice/samples"}}
        response = client.delete("/voices/test_voice")
        assert response.status_code == 200
        assert response.json()["message"] == "Voice deleted successfully"
        assert "test_voice" not in mock_tts.voices
        mock_rmtree.assert_called_once_with("voices/test_voice/samples")
        mock_yaml_dump.assert_called_once()

    @patch("talk2me.api.main.tts_engine")
    def test_delete_voice_not_found(self, mock_tts, client):
        """Test delete voice that doesn't exist."""
        mock_tts.voices = {}
        response = client.delete("/voices/nonexistent")
        assert response.status_code == 404

    @patch("talk2me.api.main.tts_engine")
    @patch("talk2me.api.main.shutil.rmtree")
    @patch("talk2me.api.main.os.path.exists", return_value=True)
    @patch("talk2me.api.main.yaml.dump", side_effect=Exception("YAML dump failed"))
    def test_delete_voice_yaml_save_failure(
        self, _mock_yaml_dump, _mock_exists, mock_rmtree, mock_tts, client
    ):
        """Test delete voice when saving config fails."""
        mock_tts.voices = {"test_voice": {"samples_dir": "voices/test_voice/samples"}}
        response = client.delete("/voices/test_voice")
        assert response.status_code == 500
        assert "Failed to save voice configuration" in response.json()["detail"]
        mock_rmtree.assert_called_once_with("voices/test_voice/samples")

    @patch("talk2me.api.main.tts_engine", None)
    def test_upload_samples_engine_not_initialized(self, client):
        """Test upload samples when TTS engine not initialized."""
        response = client.post(
            "/voices/test_voice/samples",
            files={"samples": ("sample.wav", BytesIO(b"data"), "audio/wav")},
        )
        assert response.status_code == 500

    @patch("talk2me.api.main.tts_engine")
    @patch("talk2me.api.main.os.makedirs")
    def test_upload_samples_success(self, _mock_makedirs, mock_tts, client):
        """Test successful sample upload."""
        mock_tts.voices = {"test_voice": {"samples_dir": "voices/test_voice/samples"}}
        with patch("builtins.open", mock_open()):
            response = client.post(
                "/voices/test_voice/samples",
                files={"samples": ("sample.wav", BytesIO(b"audio_data"), "audio/wav")},
            )

        assert response.status_code == 200
        result = response.json()
        assert result["uploaded"] == ["sample.wav"]
        assert "Successfully uploaded 1 samples" in result["message"]

    @patch("talk2me.api.main.tts_engine")
    def test_upload_samples_voice_not_found(self, mock_tts, client):
        """Test upload samples for non-existent voice."""
        mock_tts.voices = {}
        response = client.post(
            "/voices/nonexistent/samples",
            files={"samples": ("sample.wav", BytesIO(b"data"), "audio/wav")},
        )
        assert response.status_code == 404

    @patch("talk2me.api.main.tts_engine")
    def test_upload_samples_invalid_file(self, mock_tts, client):
        """Test upload samples with invalid file type."""
        mock_tts.voices = {"test_voice": {"samples_dir": "voices/test_voice/samples"}}
        response = client.post(
            "/voices/test_voice/samples",
            files={"samples": ("sample.mp3", BytesIO(b"data"), "audio/mp3")},
        )
        assert response.status_code == 400

    @patch("talk2me.api.main.tts_engine")
    @patch("talk2me.api.main.os.makedirs", side_effect=OSError("Permission denied"))
    def test_upload_samples_makedirs_failure(self, _mock_makedirs, mock_tts, client):
        """Test upload samples when directory creation fails."""
        mock_tts.voices = {"test_voice": {"samples_dir": "voices/test_voice/samples"}}
        response = client.post(
            "/voices/test_voice/samples",
            files={"samples": ("sample.wav", BytesIO(b"data"), "audio/wav")},
        )
        assert response.status_code == 500

    @patch("talk2me.api.main.tts_engine")
    @patch("talk2me.api.main.os.path.exists", return_value=True)
    @patch("talk2me.api.main.os.remove")
    def test_delete_sample_success(self, mock_remove, _mock_exists, mock_tts, client):
        """Test successful sample deletion."""
        mock_tts.voices = {"test_voice": {"samples_dir": "voices/test_voice/samples"}}
        response = client.delete("/voices/test_voice/samples/sample.wav")
        assert response.status_code == 200
        assert response.json()["message"] == "Sample deleted successfully"
        mock_remove.assert_called_once()

    @patch("talk2me.api.main.tts_engine")
    def test_delete_sample_voice_not_found(self, mock_tts, client):
        """Test delete sample for non-existent voice."""
        mock_tts.voices = {}
        response = client.delete("/voices/nonexistent/samples/sample.wav")
        assert response.status_code == 404

    @patch("talk2me.api.main.tts_engine")
    @patch("talk2me.api.main.os.path.exists", return_value=False)
    def test_delete_sample_not_found(self, _mock_exists, mock_tts, client):
        """Test delete sample that doesn't exist."""
        mock_tts.voices = {"test_voice": {"samples_dir": "voices/test_voice/samples"}}
        response = client.delete("/voices/test_voice/samples/nonexistent.wav")
        assert response.status_code == 404

    @patch("talk2me.api.main.tts_engine", None)
    def test_retrain_voice_engine_not_initialized(self, client):
        """Test retrain voice when TTS engine not initialized."""
        response = client.post("/voices/test_voice/retrain")
        assert response.status_code == 500

    @patch("talk2me.api.main.tts_engine")
    @patch("talk2me.api.main.Path")
    def test_retrain_voice_success(self, mock_path, mock_tts, client):
        """Test successful voice retraining."""
        mock_tts.voices = {"test_voice": {"samples_dir": "voices/test_voice/samples"}}
        mock_samples_dir = Mock()
        mock_samples_dir.glob.return_value = [Mock()]  # One sample file
        mock_path.return_value = mock_samples_dir

        response = client.post("/voices/test_voice/retrain")
        assert response.status_code == 200
        result = response.json()
        assert result["message"] == "Voice retrained successfully"
        assert result["sample_count"] == 1

    @patch("talk2me.api.main.tts_engine")
    @patch("talk2me.api.main.Path")
    def test_retrain_voice_no_samples(self, mock_path, mock_tts, client):
        """Test retrain voice with no samples."""
        mock_tts.voices = {"test_voice": {"samples_dir": "voices/test_voice/samples"}}
        mock_samples_dir = Mock()
        mock_samples_dir.glob.return_value = []  # No samples
        mock_path.return_value = mock_samples_dir

        response = client.post("/voices/test_voice/retrain")
        assert response.status_code == 400
        assert "No WAV samples found" in response.json()["detail"]

    @patch("talk2me.api.main.tts_engine")
    def test_retrain_voice_not_found(self, mock_tts, client):
        """Test retrain non-existent voice."""
        mock_tts.voices = {}
        response = client.post("/voices/nonexistent/retrain")
        assert response.status_code == 404
