"""HTTP routes for audio transcription.

Exposes the `/transcribe` endpoint, which accepts an uploaded audio or
video file, transcribes it through the shared `ASRModel` instance stored
on `app.state`, and returns a structured JSON result.
"""

import os
import uuid

from fastapi import APIRouter, UploadFile, Request, HTTPException

router = APIRouter()


@router.post("/transcribe")
async def transcribe(audio: UploadFile, request: Request):
    """Transcribe an uploaded audio or video file.

    The file is buffered to a temporary path on disk, passed to the
    `ASRModel` instance attached to the application state at startup, and
    removed in a `finally` block so the temp file does not leak when the
    response (or an error) is produced.

    Args:
        audio: Uploaded file. Only the extensions listed below are
            accepted; the check is by filename, not content sniffing.
        request: FastAPI request, used to access `app.state.asr`.

    Returns:
        A JSON object with `job_id`, detected `language`, source
        `duration` in seconds, the concatenated `full_text`, and a
        `segments` list with per-segment text, timing, and word-level
        alignment.

    Raises:
        HTTPException: 400 if the file extension is not supported.
    """
    if not audio.filename.endswith(('.wav', '.mp3', '.m4a', '.flac', '.mp4')):
        raise HTTPException(400, 'unsupported audio format')
    job_id = str(uuid.uuid4())
    tmp_path = f'/tmp/{job_id}_{audio.filename}'
    with open(tmp_path, 'wb') as buffer:
        buffer.write(await audio.read())
    try:
        asr = request.app.state.asr
        result = asr.transcribe(tmp_path)
        return {
            'job_id': job_id,
            'language': result.language,
            'duration': result.duration,
            'full_text': result.full_text,
            'segments': [s.model_dump() for s in result.segments],
        }
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
