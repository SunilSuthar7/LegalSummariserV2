from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import json
import sys
import uuid
from threading import Thread
import shutil
import subprocess
from typing import Optional
import logging

# =============== DATASET + EXTRACTION ===============
from datasets import load_dataset
import pdfplumber
import pytesseract
import docx
from odf.opendocument import load
from odf import text, teletype
# ===================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="LegalSummarizer Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = BASE_DIR / "backend" / "scripts"
SESSIONS_DIR = BASE_DIR / "backend" / "sessions"
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

PIPELINE_PROGRESS = {}

# ================= HELPERS =================

def run_script(script: Path, args: list, cwd: Path):
    cmd = [sys.executable, str(script)] + args
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr)

def save_json(path: Path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def load_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

# ================= EXTRACTION =================

def extract_text_from_pdf(path: Path) -> str:
    """Extract text from PDF with fallback to OCR"""
    text_out = ""
    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t and t.strip():
                    text_out += t + "\n"
                else:
                    try:
                        image = page.to_image(resolution=300).original
                        text_out += pytesseract.image_to_string(image) + "\n"
                    except Exception as e:
                        logger.warning(f"OCR failed for page: {e}")
                        continue
    except Exception as e:
        logger.error(f"PDF extraction failed: {e}")
        raise ValueError(f"Failed to extract text from PDF: {str(e)}")
    
    if not text_out.strip():
        raise ValueError("No text could be extracted from PDF")
    return text_out.strip()

def extract_text_from_docx(path: Path) -> str:
    """Extract text from DOCX"""
    try:
        doc = docx.Document(path)
        text_data = "\n".join(p.text for p in doc.paragraphs).strip()
        if not text_data:
            raise ValueError("DOCX file is empty")
        return text_data
    except Exception as e:
        raise ValueError(f"Failed to extract text from DOCX: {str(e)}")

def extract_text_from_odt(path: Path) -> str:
    """Extract text from ODT"""
    try:
        doc = load(str(path))
        paras = doc.getElementsByType(text.P)
        text_data = "\n".join(teletype.extractText(p) for p in paras).strip()
        if not text_data:
            raise ValueError("ODT file is empty")
        return text_data
    except Exception as e:
        raise ValueError(f"Failed to extract text from ODT: {str(e)}")

def extract_text_from_txt(path: Path) -> str:
    """Extract text from TXT"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            text_data = f.read().strip()
        if not text_data:
            raise ValueError("TXT file is empty")
        return text_data
    except Exception as e:
        raise ValueError(f"Failed to extract text from TXT: {str(e)}")

# ================= ROUTES =================

@app.get("/")
def root():
    return {"message": "Backend running", "sessions": len(PIPELINE_PROGRESS)}

@app.post("/run_pipeline")
async def run_pipeline(
    mode: str = Form(...),
    dataset: Optional[str] = Form(None),
    n: Optional[int] = Form(None),
    file: UploadFile = File(None),
):
    session_id = str(uuid.uuid4())
    session_path = SESSIONS_DIR / session_id
    session_path.mkdir(parents=True, exist_ok=True)

    # Initialize session IMMEDIATELY before any processing
    PIPELINE_PROGRESS[session_id] = {
        "stages": [],
        "completed": False,
        "results": None,
        "error": None
    }
    
    logger.info(f"[{session_id}] Session initialized - Mode: {mode}")

    def update(stage):
        if session_id in PIPELINE_PROGRESS:
            PIPELINE_PROGRESS[session_id]["stages"].append(stage)
            logger.info(f"[{session_id}] Stage: {stage}")

    def pipeline_task():
        try:
            raw_samples = []

            # ================= UPLOAD MODE =================
            if mode == "upload":
                if not file:
                    raise ValueError("File required for upload mode")

                logger.info(f"[{session_id}] Processing uploaded file: {file.filename}")
                
                file_path = session_path / file.filename
                with open(file_path, "wb") as f:
                    shutil.copyfileobj(file.file, f)

                logger.info(f"[{session_id}] File saved to {file_path}")

                # Extract text from uploaded file
                ext = file.filename.lower().split(".")[-1]
                logger.info(f"[{session_id}] File extension: {ext}")
                
                try:
                    if ext == "pdf":
                        logger.info(f"[{session_id}] Extracting text from PDF...")
                        text_data = extract_text_from_pdf(file_path)
                    elif ext == "docx":
                        logger.info(f"[{session_id}] Extracting text from DOCX...")
                        text_data = extract_text_from_docx(file_path)
                    elif ext == "odt":
                        logger.info(f"[{session_id}] Extracting text from ODT...")
                        text_data = extract_text_from_odt(file_path)
                    elif ext == "txt":
                        logger.info(f"[{session_id}] Extracting text from TXT...")
                        text_data = extract_text_from_txt(file_path)
                    else:
                        raise ValueError(f"Unsupported file type: .{ext}")
                    
                    logger.info(f"[{session_id}] Text extraction successful - Length: {len(text_data)} chars")
                except Exception as e:
                    logger.error(f"[{session_id}] Text extraction failed: {str(e)}")
                    raise

                raw_samples.append({
                    "id": session_id,
                    "input_text": text_data
                })

            # ================= DATASET MODE =================
            elif mode == "dataset":
                if not dataset or not n:
                    raise ValueError("Dataset name and n required")

                update("Dataset Loading")
                logger.info(f"[{session_id}] Loading {dataset} dataset with {n} samples")

                try:
                    if dataset == "ILC":
                        logger.info(f"[{session_id}] Loading ILC dataset...")
                        ds = load_dataset("d0r1h/ILC", split="train[:{}]".format(n))
                        for i, r in enumerate(ds):
                            raw_samples.append({
                                "id": f"ilc_{i}",
                                "input_text": r.get("Case", "")
                            })
                        logger.info(f"[{session_id}] ILC dataset loaded - {len(raw_samples)} samples")

                    elif dataset == "IN-ABS":
                        logger.info(f"[{session_id}] Loading IN-ABS dataset...")
                        ds = load_dataset("percins/IN-ABS", split="train[:{}]".format(n))
                        for i, r in enumerate(ds):
                            raw_samples.append({
                                "id": f"inabs_{i}",
                                "input_text": r.get("text", "")
                            })
                        logger.info(f"[{session_id}] IN-ABS dataset loaded - {len(raw_samples)} samples")

                    else:
                        raise ValueError("Unsupported dataset")
                except Exception as e:
                    raise ValueError(f"Failed to load dataset: {str(e)}")

            else:
                raise ValueError("Invalid mode")

            # ================= COMMON PIPELINE =================
            logger.info(f"[{session_id}] Starting pipeline with {len(raw_samples)} samples")
            save_json(session_path / "raw.json", raw_samples)

            update("Cleaning")
            logger.info(f"[{session_id}] Running cleaner script...")
            run_script(
                SCRIPTS_DIR / "cleaner_generic.py",
                ["--input", "raw.json", "--output", "cleaned.json"],
                session_path
            )

            update("LegalBERT Extractive")
            logger.info(f"[{session_id}] Running LegalBERT extractive script...")
            run_script(
                SCRIPTS_DIR / "legalbert_extractive.py",
                ["--input", "cleaned.json", "--output", "legalbert.json"],
                session_path
            )

            update("T5 Abstractive")
            logger.info(f"[{session_id}] Running T5 abstractive script...")
            run_script(
                SCRIPTS_DIR / "t5_abstractive.py",
                ["--input", "legalbert.json", "--output", "final.json"],
                session_path
            )

            logger.info(f"[{session_id}] Loading results...")
            if session_id in PIPELINE_PROGRESS:
                PIPELINE_PROGRESS[session_id]["results"] = load_json(
                    session_path / "final.json"
                )
                PIPELINE_PROGRESS[session_id]["completed"] = True
                logger.info(f"[{session_id}] Pipeline completed successfully")

        except Exception as e:
            logger.error(f"[{session_id}] Pipeline error: {str(e)}")
            if session_id in PIPELINE_PROGRESS:
                PIPELINE_PROGRESS[session_id]["error"] = str(e)
                PIPELINE_PROGRESS[session_id]["completed"] = True

    # Start pipeline in background thread
    Thread(target=pipeline_task, daemon=True).start()
    
    logger.info(f"[{session_id}] Returning session_id to client")
    return {"status": "started", "session_id": session_id}

@app.get("/pipeline_status")
def pipeline_status(session_id: str):
    if session_id not in PIPELINE_PROGRESS:
        logger.warning(f"Session not found: {session_id}")
        logger.info(f"Available sessions: {list(PIPELINE_PROGRESS.keys())}")
        raise HTTPException(404, f"Session '{session_id}' not found")
    
    return PIPELINE_PROGRESS[session_id]
