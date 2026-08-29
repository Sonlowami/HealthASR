"""
HealthASR Whisper transcription API (Swagger UI at /docs).

Modes (env):
  MODEL_URL   — if set, POST /transcribe proxies audio to this remote ASR base URL
                (expects remote POST {MODEL_URL}/v1/asr with multipart file + language)
  MODEL_PATH  — local HF Whisper `final/` OR a QAT `quantized/` dir
                (folder with `quantized_state_dict.pt`)

  Optional:
  DEVICE=cuda|cpu|auto   (default auto)
  BASE_CONFIG=openai/whisper-large-v3
      — HF config id used only when loading int8 QAT (architecture skeleton;
        quantized weights come from MODEL_PATH; no full FP checkpoint needed)
  QAT_SCHEME=int8_weight_qat
"""
from __future__ import annotations

import io
import os
import sys
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

# Repo root on sys.path so training.whisper.compression imports work when
# launched as `uvicorn serving.app:app --app-dir .`
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

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
        "Set MODEL_URL (proxy to remote ASR) or MODEL_PATH (local Whisper "
        "final/ or int8 quantized/ folder)."
    )


def _is_quantized_dir(path: Path) -> bool:
    return (path / "quantized_state_dict.pt").is_file()


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
        audio, sr = sf.read(io.BytesIO(data), always_2d=False)
        if getattr(audio, "ndim", 1) > 1:
            audio = np.mean(audio, axis=-1)
        audio = audio.astype(np.float32)
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        return audio, 16000
    except Exception:
        pass

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
        from transformers import (
            GenerationConfig,
            WhisperConfig,
            WhisperForConditionalGeneration,
            WhisperProcessor,
        )

        self.device = device
        self.model_path = model_path
        self._torch = torch
        path = Path(model_path)

        if _is_quantized_dir(path):
            self.model, self.processor = self._load_int8_qat(path, device)
            print(f"[serving] loaded int8 QAT from {path}", flush=True)
        else:
            self.processor = WhisperProcessor.from_pretrained(model_path)
            self.model = WhisperForConditionalGeneration.from_pretrained(model_path)
            if not getattr(self.model.generation_config, "lang_to_id", None):
                self.model.generation_config = GenerationConfig.from_pretrained(
                    "openai/whisper-large-v3"
                )
            self.model.generation_config.forced_decoder_ids = None
            self.model.to(device)
            self.model.eval()

    def _load_int8_qat(self, quantized_dir: Path, device: str):
        """Rebuild torchao int8 weight-only model from quantized_state_dict.pt."""
        import torch
        from transformers import (
            GenerationConfig,
            WhisperConfig,
            WhisperForConditionalGeneration,
            WhisperProcessor,
        )

        try:
            from training.whisper.compression.quantize import get_qat_scheme, quantize_model
        except ImportError as e:
            raise RuntimeError(
                "Could not import training.whisper.compression.quantize. "
                "Run uvicorn from the HealthASR repo root with --app-dir ."
            ) from e

        scheme = os.environ.get("QAT_SCHEME", "int8_weight_qat").strip()
        base_config = os.environ.get("BASE_CONFIG", "openai/whisper-large-v3").strip()
        sd_path = quantized_dir / "quantized_state_dict.pt"

        # Processor was saved next to the state dict; config is large-v3 / SALT-shaped
        processor = WhisperProcessor.from_pretrained(str(quantized_dir))
        config = WhisperConfig.from_pretrained(base_config)
        model = WhisperForConditionalGeneration(config)
        if not getattr(model.generation_config, "lang_to_id", None):
            model.generation_config = GenerationConfig.from_pretrained(base_config)
        model.generation_config.forced_decoder_ids = None

        quantize_model(model, config=get_qat_scheme(scheme)["base"])
        try:
            state = torch.load(sd_path, map_location=device, weights_only=True)
        except TypeError:
            state = torch.load(sd_path, map_location=device)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"[serving] load_state_dict missing keys: {len(missing)}", flush=True)
        if unexpected:
            print(f"[serving] load_state_dict unexpected keys: {len(unexpected)}", flush=True)

        model.to(device)
        model.eval()
        return model, processor

    def transcribe(self, audio: np.ndarray, language_key: str, max_new_tokens: int = 128) -> str:
        token_id = LANG_TOKEN_IDS.get(language_key.lower())
        if token_id is None:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown language '{language_key}'. Use: kin, dav (or kinyarwanda, kidawida).",
            )
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
        kind = "int8-qat" if _is_quantized_dir(Path(path)) else "hf-final"
        print(f"[serving] local ({kind}) → {path} on {device}", flush=True)
    yield
    _state.clear()


app = FastAPI(
    title="HealthASR Whisper API",
    description=(
        "Transcribe Kinyarwanda (`kin`) or Kidaw'ida (`dav`) audio.\n\n"
        "- **Local HF:** `MODEL_PATH` = Whisper `final/` checkpoint.\n"
        "- **Local int8 QAT:** `MODEL_PATH` = extracted `quantized/` folder "
        "(contains `quantized_state_dict.pt`).\n"
        "- **Proxy:** `MODEL_URL` → remote `POST /v1/asr`.\n\n"
        "Open **Swagger UI** at `/docs`."
    ),
    version="0.2.0",
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
