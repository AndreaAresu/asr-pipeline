from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.config import settings
from app.db.models import Base


engine = create_engine(settings.database_url, echo=False)
SessionLocal = sessionmaker(engine, autocommit=False, autoflush=False)

def init_db():
    Base.metadata.create_all(bind=engine)
