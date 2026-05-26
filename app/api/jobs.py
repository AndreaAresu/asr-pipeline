from fastapi import APIRouter, HTTPException
from app.db.session import SessionLocal
from app.db.models import Job, Transcript


router = APIRouter()

@router.get("/jobs/{job_id}")
def get_job(job_id: str):
    db = SessionLocal()
    job = db.get(Job, job_id)
    db.close()

    if job is None:
        raise HTTPException(404, "job not found")
    return {
        "id": job.id,
        "status": job.status,
        "error_message": job.error_message,
        "duration": job.duration,
    }

@router.get("/jobs/{job_id}/result")
def get_result(job_id: str):
    db = SessionLocal()
    job = db.get(Job, job_id)
    if job is None:
        db.close()
        raise HTTPException(404, "job not found")
    if job.status != "done":
        db.close()
        raise HTTPException(400, f"job not completed: status={job.status}")
    t = db.query(Transcript).filter_by(job_id=job_id).first()
    db.close()
    return {
        "full_text": t.full_text,
        "language": t.language,
        "segments": t.word_timestamps,
    }