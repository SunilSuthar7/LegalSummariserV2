from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import json
import sys
from typing import Optional
import time
import uuid
from threading import Thread
import shutil
import subprocess
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="LegalSummarizer Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SCRIPTS_DIR = BASE_DIR / "backend" / "scripts"
SESSIONS_DIR = BASE_DIR / "backend" / "sessions"
DATA_DIR.mkdir(parents=True, exist_ok=True)
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/sessions", StaticFiles(directory=SESSIONS_DIR), name="sessions")

# Track pipelines by session_id
PIPELINE_PROGRESS = {}

@app.get("/")
def root():
    return {"message": "Backend is running! Use POST /run_pipeline"}

def safe_load_json(path: Path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def save_json(path: Path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def run_script(script_path: Path, args: list = [], verbose=False, stage_name=None):
    cmd = [sys.executable, str(script_path)] + args
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Script failed: {' '.join(cmd)}\nExit code: {result.returncode}\n\n"
            f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
        )
    return result.stdout

@app.post("/run_pipeline")
async def run_pipeline(dataset: str = Form(...), n: int = Form(...), entry_id: Optional[int] = Form(None)):
    dataset = dataset.strip().upper()
    if dataset not in ["ILC", "IN-ABS"]:
        raise HTTPException(status_code=422, detail="Dataset must be 'ILC' or 'IN-ABS'")
    if n <= 0 or n > 1000:
        raise HTTPException(status_code=422, detail="n must be between 1 and 1000")

    session_id = str(uuid.uuid4())
    PIPELINE_PROGRESS[session_id] = {
        "stages": [],
        "completed": False,
        "results": None,
        "error": None,
        "entry_id": entry_id,
    }

    def pipeline_task():
        try:
            outputs = {}
            stages = []

            def update_stage(name, status):
                stage_entry = {"stage": name, "status": status}
                stages.append(stage_entry)
                PIPELINE_PROGRESS[session_id]["stages"] = stages.copy()

            # Prepare session folder
            session_path = SESSIONS_DIR / session_id
            session_path.mkdir(parents=True, exist_ok=True)

            # Step 1: Load cleaned/chunked data
            stage_name = "Cleaning"
            update_stage(stage_name, "completed")  # mark cleaning done

            if dataset == "ILC":
                cleaned_path = DATA_DIR / "cleaned_ilc.json"
                chunked_path = DATA_DIR / "chunked_ilc.json"
                cleaned_data = safe_load_json(cleaned_path) or []
                chunked_data = safe_load_json(chunked_path) or []
            else:
                cleaned_path = DATA_DIR / "cleaned_inabs.json"
                chunked_path = DATA_DIR / "chunked_inabs.json"
                cleaned_data = safe_load_json(cleaned_path) or []
                chunked_data = safe_load_json(chunked_path) or []

            update_stage("Chunking", "completed")

            # Extract subset
            if entry_id:
                cleaned_subset = [d for d in cleaned_data if d.get("id") == entry_id]
                chunked_subset = [d for d in chunked_data if d.get("id") == entry_id] if chunked_data else None
            else:
                cleaned_subset = cleaned_data[:n]
                chunked_subset = chunked_data[:n] if chunked_data else None

            # Save session copies
            save_json(session_path / "cleaned.json", cleaned_subset)
            if chunked_subset:
                save_json(session_path / "chunked.json", chunked_subset)
            outputs["cleaned"] = str(session_path / "cleaned.json")
            if chunked_subset:
                outputs["chunked"] = str(session_path / "chunked.json")
            update_stage(stage_name, "completed")

            # Step 2: Summarization
            stage_name = "Summarization"
            run_script(
                SCRIPTS_DIR / "t5_summarizer.py",
                [
                    "--dataset", dataset,
                    "--session_id", session_id,
                    "--n", str(n)
                ] + (["--ids", str(entry_id)] if entry_id else []),
            )
            outputs["summary"] = str(session_path / f"t5_{dataset.lower()}_final.json")
            update_stage(stage_name, "completed")

            # Step 3: Evaluation
            stage_name = "Evaluation"
            run_script(
                SCRIPTS_DIR / "t5_evaluator.py",
                [
                    "--dataset", dataset,
                    "--session_id", session_id,
                    "--n", str(n)
                ] + (["--ids", str(entry_id)] if entry_id else []),
            )
            outputs["evaluation"] = str(session_path / f"rouge_{dataset.lower()}.json")
            update_stage(stage_name, "completed")

            # Prepare final entries for frontend
            summary_json = safe_load_json(session_path / f"t5_{dataset.lower()}_final.json") or []
            eval_json = safe_load_json(session_path / f"rouge_{dataset.lower()}.json") or {}
            per_entry_rouge = eval_json.get("per_entry", [])

            entries = []
            for i, s in enumerate(summary_json):
                c = cleaned_subset[i] if i < len(cleaned_subset) else {}
                rouge_entry = per_entry_rouge[i] if i < len(per_entry_rouge) else None
                entries.append({
                    "original_text": c.get("input_text"),
                    "reference_summary": c.get("summary_text"),
                    "generated_summary": s.get("refined_summary_improved"),
                    "rouge_scores": rouge_entry,
                })

            avg_rouge = {k: eval_json.get(k) for k in ["rouge1", "rouge2", "rougeL"]}

            PIPELINE_PROGRESS[session_id]["results"] = {
                "avg_scores": avg_rouge,
                "entries": entries,
                "outputs": outputs,
            }
            PIPELINE_PROGRESS[session_id]["completed"] = True

        except Exception as e:
            PIPELINE_PROGRESS[session_id]["error"] = str(e)
            PIPELINE_PROGRESS[session_id]["completed"] = True

    Thread(target=pipeline_task).start()
    return {"status": "started", "session_id": session_id}

@app.get("/pipeline_status")
def pipeline_status(session_id: str):
    if session_id in PIPELINE_PROGRESS:
        return PIPELINE_PROGRESS[session_id]
    else:
        raise HTTPException(status_code=404, detail="Session not found")
