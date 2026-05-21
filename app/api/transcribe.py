import os, uuid
from fastapi import APIRouter, UploadFile, File, Request, HTTPException

router = APIRouter()


@router.post("/transcribe")
async def transcribe(audio: UploadFile, request: Request):
    if not audio.filename.endswith(('.wav', '.mp3', '.m4a', '.flac', ".mp4")):
        raise HTTPException(400, 'unsupported audio format')
    job_id = str(uuid.uuid4()) # generate random uuid
    tmp_path = f'/tmp/{job_id}_{audio.filename}'
    with open (tmp_path, 'wb') as buffer:
        buffer.write(await audio.read())
    try:
        asr = request.app.state.asr
        result = asr.transcribe(tmp_path)
        return {
            'job_id': job_id,
            'language': result.language,
            'duration': result.duration,
            'full_text': result.full_text,
            'segments': [s.model_dump() for s in result.segments]
        }
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)