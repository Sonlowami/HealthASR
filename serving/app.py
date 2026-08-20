"""
HealthASR Whisper transcription API (Swagger UI at /docs).

Modes (env):
  MODEL_URL   — if set, POST /transcribe proxies audio to this remote ASR base URL
                (expects remote POST {MODEL_URL}/v1/asr with multipart file + language)
  MODEL_PATH  — local HF Whisper checkpoint dir (used when MODEL_URL is unset)

  Optional:
  DEVICE=cuda|cpu|auto   (default auto)
  HOST=0.0.0.0 PORT=8000
"""
from __future__ import annotations

import io
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

# User-facing language → Whisper/SALT lang token id (same as config/*.yaml)
LANG_TOKEN_IDS = {
    "kin": 50350,  # Kinyarwanda <|kin|>
    "kinyarwanda": 50350,
    "dav": 50318,  # Kidaw'ida via <|sw|> / swa proxy
    "kidawida": 50318,
    "sw": 50318,
    "swa": 50318,
}


class TranscribeResponse(BaseModel):
    text: str
    language: str
    mode: Literal["local", "proxy"]
    model: str | None = None


class HealthResponse(BaseModel):
    status: str = "ok"
    mode: Literal["local", "proxy"]
    model: str | None = None
    device: str | None = None


def _resolve_mode() -> Literal["local", "proxy"]:
    if os.environ.get("MODEL_URL", "").strip():
        return "proxy"
    if os.environ.get("MODEL_PATH", "").strip():
        return "local"
    raise RuntimeError(
        "Set MODEL_URL (proxy to remote ASR) or MODEL_PATH (local Whisper checkpoint)."
    )


def _load_audio_bytes(data: bytes, filename: str | None = None) -> tuple[np.ndarray, int]:
    """Decode upload to mono float32 waveform at 16 kHz."""
    try:
        import librosa
        import soundfile as sf
    except ImportError as e:
        raise HTTPException(
            status_code=500,
            detail="librosa and soundfile are required for local audio decode. "
            "pip install -r serving/requirements.txt",
        ) from e

    suffix = Path(filename or "audio.wav").suffix or ".wav"
    try:
        # soundfile first (wav/flac)
        audio, sr = sf.read(io.BytesIO(data), always_2d=False)
        if getattr(audio, "ndim", 1) > 1:
            audio = np.mean(audio, axis=-1)
        audio = audio.astype(np.float32)
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        return audio, 16000
    except Exception:
        pass

    # fallback via temp file (mp3/webm/etc.)
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        audio, sr = librosa.load(tmp_path, sr=16000, mono=True)
        return audio.astype(np.float32), 16000
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not decode audio: {e}") from e
    finally:
        Path(tmp_path).unlink(missing_ok=True)


class LocalWhisper:
    def __init__(self, model_path: str, device: str):
        import torch
        from transformers import GenerationConfig, WhisperForConditionalGeneration, WhisperProcessor

        self.device = device
        self.model_path = model_path
        self.processor = WhisperProcessor.from_pretrained(model_path)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_path)
        if not getattr(self.model.generation_config, "lang_to_id", None):
            self.model.generation_config = GenerationConfig.from_pretrained(
                "openai/whisper-large-v3"
            )
        self.model.generation_config.forced_decoder_ids = None
        self.model.to(device)
        self.model.eval()
        self._torch = torch

    def transcribe(self, audio: np.ndarray, language_key: str, max_new_tokens: int = 128) -> str:
        token_id = LANG_TOKEN_IDS.get(language_key.lower())
        if token_id is None:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown language '{language_key}'. Use: kin, dav (or kinyarwanda, kidawida).",
            )
        # Match train.py: decode lang token id → string for generate(language=...)
        language = self.processor.tokenizer.decode([token_id])
        inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt")
        input_features = inputs.input_features.to(self.device)
        with self._torch.inference_mode():
            predicted = self.model.generate(
                input_features,
                language=language,
                task="transcribe",
                max_new_tokens=max_new_tokens,
            )
        text = self.processor.batch_decode(predicted, skip_special_tokens=True)[0]
        return text.strip()


class ProxyClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def transcribe(self, data: bytes, filename: str, content_type: str | None, language: str) -> str:
        try:
            import httpx
        except ImportError as e:
            raise HTTPException(
                status_code=500,
                detail="httpx required for proxy mode. pip install httpx",
            ) from e

        url = f"{self.base_url}/v1/asr"
        files = {"file": (filename or "audio.wav", data, content_type or "application/octet-stream")}
        form = {"language": language}
        try:
            with httpx.Client(timeout=180.0) as client:
                r = client.post(url, files=files, data=form)
        except httpx.RequestError as e:
            raise HTTPException(status_code=502, detail=f"Remote MODEL_URL unreachable: {e}") from e
        if r.status_code >= 400:
            raise HTTPException(
                status_code=502,
                detail=f"Remote ASR error {r.status_code}: {r.text[:500]}",
            )
        body = r.json()
        if isinstance(body, dict) and "text" in body:
            return str(body["text"])
        raise HTTPException(status_code=502, detail=f"Unexpected remote response: {body!r}")


_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    mode = _resolve_mode()
    _state["mode"] = mode
    if mode == "proxy":
        url = os.environ["MODEL_URL"].strip()
        _state["backend"] = ProxyClient(url)
        _state["model"] = url
        _state["device"] = None
        print(f"[serving] proxy mode → {url}", flush=True)
    else:
        import torch

        path = os.environ["MODEL_PATH"].strip()
        if not Path(path).exists():
            raise RuntimeError(f"MODEL_PATH not found: {path}")
        dev_env = os.environ.get("DEVICE", "auto").lower()
        if dev_env == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = dev_env
        _state["backend"] = LocalWhisper(path, device)
        _state["model"] = path
        _state["device"] = device
        print(f"[serving] local mode → {path} on {device}", flush=True)
    yield
    _state.clear()


app = FastAPI(
    title="HealthASR Whisper API",
    description=(
        "Transcribe Kinyarwanda (`kin`) or Kidaw'ida (`dav`) audio.\n\n"
        "- **Local mode:** set `MODEL_PATH` to an HF Whisper `final/` checkpoint.\n"
        "- **Proxy mode:** set `MODEL_URL` to a remote ASR base that implements "
        "`POST /v1/asr` (multipart `file` + form `language`).\n\n"
        "Open **Swagger UI** at `/docs`."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health():
    return HealthResponse(
        status="ok",
        mode=_state.get("mode", "local"),
        model=_state.get("model"),
        device=_state.get("device"),
    )


@app.post(
    "/transcribe",
    response_model=TranscribeResponse,
    tags=["asr"],
    summary="Transcribe an audio file",
)
async def transcribe(
    file: UploadFile = File(..., description="Audio file (wav/mp3/flac/…)"),
    language: str = Form(
        "kin",
        description="Language: kin / kinyarwanda or dav / kidawida",
    ),
):
    """Upload audio and get a transcript. Uses local Whisper or proxies to MODEL_URL."""
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file upload")

    mode = _state["mode"]
    backend = _state["backend"]

    if mode == "proxy":
        text = backend.transcribe(
            data,
            filename=file.filename or "audio.wav",
            content_type=file.content_type,
            language=language,
        )
    else:
        audio, _ = _load_audio_bytes(data, file.filename)
        if audio.size == 0:
            raise HTTPException(status_code=400, detail="Decoded audio is empty")
        text = backend.transcribe(audio, language_key=language)

    return TranscribeResponse(
        text=text,
        language=language,
        mode=mode,
        model=_state.get("model"),
    )


@app.post(
    "/v1/asr",
    response_model=TranscribeResponse,
    tags=["asr"],
    summary="Same as /transcribe (stable path for remote clients)",
    include_in_schema=True,
)
async def v1_asr(
    file: UploadFile = File(...),
    language: str = Form("kin"),
):
    """Alias so a local server can also act as MODEL_URL for another Swagger proxy."""
    return await transcribe(file=file, language=language)
