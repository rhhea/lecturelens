"""
main.py
-------
FastAPI service for LectureLens. One endpoint takes an uploaded file
and returns:
  - a summary
  - follow-up questions
  - recommended resources

Run locally with:
    uvicorn main:app --reload --port 8000

Then your Node/Express backend calls this service, e.g.:
    POST http://localhost:8000/process   (multipart file upload)
"""

import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ingestion import process_file
from summarizer import summarize_document
from question_gen import generate_followup_questions
from recommender import recommend_resources

app = FastAPI(title="LectureLens NLP Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ProcessResponse(BaseModel):
    summary: str
    followup_questions: list[str]
    recommended_resources: list[dict]
    passage_count: int


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/process", response_model=ProcessResponse)
async def process_notes(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".pdf", ".docx", ".txt"}:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")

    # Save the upload to a temp file so our extractors (which expect a
    # file path) can read it.
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        passages = process_file(tmp_path)
        if not passages:
            raise HTTPException(status_code=422, detail="No extractable text found in file.")

        summary = summarize_document(passages)

        # Generate follow-ups from the first passage for now -- once this
        # works end-to-end, consider generating from each passage and
        # deduping, so long documents get broader question coverage.
        questions = generate_followup_questions(passages[0])

        resources = recommend_resources(" ".join(passages[:2]))  # use first ~2 passages as topic signal

        return ProcessResponse(
            summary=summary,
            followup_questions=questions,
            recommended_resources=resources,
            passage_count=len(passages),
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)