# HealthASR — Whisper Swagger API

FastAPI app with interactive docs at **`/docs`**.

## Modes

| Env | Mode | Behavior |
|-----|------|----------|
| `MODEL_PATH` | **local** | Load HF Whisper checkpoint; decode audio → transcribe |
| `MODEL_URL` | **proxy** | Forward upload to `{MODEL_URL}/v1/asr` (remote deployed model) |

If both are set, **`MODEL_URL` wins** (proxy).

Languages: `kin` / `kinyarwanda`, `dav` / `kidawida` (Kidaw'ida uses the SALT `sw` token).

## Setup

```bash
conda activate healthasr   # or any env with torch + transformers
pip install -r serving/requirements.txt
```

## Run (local model)

```bash
export MODEL_PATH=/project/community/rmwisene/pipeline_outputs/whisper_runs/kin-dav-balanced-27h-curriculum-e15/final
# optional: DEVICE=cuda|cpu|auto
uvicorn serving.app:app --host 0.0.0.0 --port 8000 --app-dir .
```

Open: [http://localhost:8000/docs](http://localhost:8000/docs)

- Try **POST /transcribe** → upload a wav/mp3 → set `language` to `kin` or `dav` → Execute.

## Run (proxy to remote deployed model)

Remote server must implement:

`POST {MODEL_URL}/v1/asr`  
multipart: `file` (audio), form: `language`  
response JSON: `{"text": "..."}`  

(This app’s own `/v1/asr` matches that contract, so one GPU box can host the model and another process can proxy to it.)

```bash
export MODEL_URL=https://your-gpu-host.example.com
uvicorn serving.app:app --host 0.0.0.0 --port 8000 --app-dir .
```

Then open `/docs` and call `/transcribe` the same way — the Swagger UI only talks to this API; this API calls the remote model.

## Health check

`GET /health` → `{"status":"ok","mode":"local"|"proxy", ...}`

## Orchard note

Prefer a **GPU compute node** (or cloud GPU) for `MODEL_PATH` mode. Login nodes are for light work only. For a public demo URL, deploy on a GPU VM / HF Space and set `MODEL_PATH` or point a lightweight `/docs` instance at `MODEL_URL`.
