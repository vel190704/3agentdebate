from fastapi import FastAPI

from . import models  # noqa: F401  (registers ORM models on Base before create_all)
from .db import Base, engine
from .routers import board, tasks

Base.metadata.create_all(bind=engine)

app = FastAPI(title="Multi-LLM Consensus Platform")
app.include_router(tasks.router)
app.include_router(board.router)


@app.get("/health")
def health():
    return {"status": "ok"}
