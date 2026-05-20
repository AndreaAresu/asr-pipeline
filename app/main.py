from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.core import ASRModel
from app.config import settings


@asynccontextmanager
async def lifespan(app:FastAPI):
    app.state.asr = ASRModel(settings.whisper_model)
    print("startup")
    
    yield

    print("shutdown")
    
app = FastAPI(lifespan=lifespan, title="ASR Pipeline", version="0.1.0")

@app.get('/health')
async def health():
    return {'status':'ok'} 