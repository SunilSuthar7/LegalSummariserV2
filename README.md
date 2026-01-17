
# LegalSummarizer

**LegalSummarizer** is an end-to-end web-based AI system for summarizing **Indian legal documents** using a **hybrid extractive–abstractive approach**.  
The system combines **LegalBERT-based extractive summarization** with **QLoRA fine-tuned T5 abstractive summarization**, enabling high-quality summaries of long legal judgments under **low-resource constraints**.

It supports both benchmark datasets (**ILC** and **IN-ABS**) as well as **user-uploaded legal documents (PDF/DOCX)** and provides **ROUGE-based evaluation**.

---

## Project Highlights

- Hybrid **LegalBERT + T5** summarization pipeline  
- **QLoRA (4-bit + LoRA)** fine-tuning for memory-efficient training  
- Handles **long legal documents** using hierarchical (two-stage) summarization  
- Dataset-based and **user-uploaded document** summarization  
- End-to-end **FastAPI backend + web frontend**  
- Achieves **ROUGE-1 ≈ 47.3**, outperforming prior reported results (~46)

---

## Features

- Cleaning and normalization of Indian legal texts  
- Sentence-level **extractive summarization using fine-tuned LegalBERT**  
- Two-stage **abstractive summarization using QLoRA-enhanced T5**  
- Chunk-safe summarization for documents exceeding model token limits  
- Support for **ILC** and **IN-ABS** datasets  
- Upload and summarize custom legal documents (PDF/DOCX/ODT)  
- ROUGE-1, ROUGE-2, ROUGE-L evaluation  
- Session-based execution with real-time progress tracking  
- Modular, reproducible, and scalable architecture  

---

## System Architecture

1. **Input Selection**
   - ILC dataset
   - IN-ABS dataset
   - User-uploaded legal document

2. **Text Extraction**
   - `pdfplumber` for digital PDFs  
   - `pytesseract` OCR fallback for scanned documents  

3. **Preprocessing**
   - Cleaning, normalization, sentence segmentation  

4. **Extractive Stage**
   - Fine-tuned **LegalBERT classifier**
   - Filters and ranks legally relevant sentences  

5. **Abstractive Stage**
   - **T5-base** with **QLoRA adapters**
   - Chunk-level summaries followed by global refinement  

6. **Evaluation**
   - ROUGE score computation  

7. **Frontend Display**
   - Original text
   - Reference summary
   - Generated summary
   - ROUGE metrics  

---

## Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/SunilSuthar7/LegalSummarizerV2.git
cd LegalSummarizerV2
````

---

### 2️⃣ Create and Activate Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Application

### 1️⃣ Start the Backend (FastAPI)

```bash
python backend/run_server.py
```

* Backend runs at: `http://127.0.0.1:8000`
* Uses session-based execution for long-running pipelines
* Exposes REST APIs for:

  * Dataset selection
  * File upload
  * Pipeline execution
  * Progress polling
  * Result retrieval

---

### 2️⃣ Start the Frontend

The frontend is built using **HTML, CSS, and vanilla JavaScript**.

#### Option 1: Open Directly

* Navigate to the `frontend/` folder
* Open `index.html` in a browser

#### Option 2: Serve via HTTP Server (Recommended)

```bash
cd frontend
python -m http.server 5500
```

Open in browser:

```text
http://localhost:5500/index.html
```

---

## Using the Application

1. **Select Input Source**

   * ILC dataset
   * IN-ABS dataset
   * Upload a legal document (PDF/DOCX)

2. **Configure Run**

   * Choose dataset entries or specific IDs
   * Upload document if applicable

3. **Run Pipeline**

   * Click **Run Pipeline**
   * Monitor real-time progress:

     * Text extraction
     * Cleaning
     * Extractive summarization
     * Abstractive summarization
     * Evaluation

4. **View Results**

   * Original legal text
   * Reference summary (for datasets)
   * AI-generated summary
   * ROUGE scores

---

## Evaluation Results (Validation Set)

| Model                  | ROUGE-1   | ROUGE-2   | ROUGE-L   |
| ---------------------- | --------- | --------- | --------- |
| Base T5                | ~39       | ~18       | ~19       |
| T5 + QLoRA             | ~42       | ~20       | ~21       |
| LegalBERT (Extractive) | ~26       | ~10       | ~15       |
| **Hybrid (Proposed)**  | **47.32** | **22.84** | **23.72** |

---

## Technology Stack

* **Backend:** Python, FastAPI, PyTorch
* **Models:** LegalBERT, T5-base, QLoRA (PEFT)
* **Frontend:** HTML, CSS, JavaScript
* **Datasets:** ILC, IN-ABS
* **Evaluation:** ROUGE
* **Deployment:** Local machine, Google Colab (free tier)

---

## Notes

* Designed to run on **RTX 3050 (4GB VRAM)** and **free Google Colab**
* QLoRA enables fine-tuning without full model retraining
* Modular design allows easy extension to other legal datasets
* Suitable for academic projects, demos, and research experiments

---





