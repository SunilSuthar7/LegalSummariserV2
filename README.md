
# LegalSummarizer

LegalSummarizer is a web application for summarizing Indian legal documents using **T5-based abstractive summarization**. It supports **ILC (Indian Legal Cases)** and **IN-ABS (Indian Abstractive Summaries)** datasets and provides ROUGE evaluation for the generated summaries.

---

## Features

- Clean and preprocess legal documents.  
- Generate abstractive summaries using T5.  
- Evaluate summaries using ROUGE metrics.  
- View detailed comparison: original text, reference summary, AI-generated summary.  
- Supports both **ILC** and **IN-ABS** datasets.

---

## Installation

1. **Clone the repository**

```bash
git clone https://github.com/SunilSuthar7/LegalSummarizerV2.git
cd LegalSummarizer
````

2. **Create a virtual environment and activate it**

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

---

## Running the Application

### 1️⃣ Start the Backend

```bash
python backend/run_server.py
```

* FastAPI backend runs at `http://127.0.0.1:8000`.
* Provides endpoints to run the summarization pipeline and fetch session results.

### 2️⃣ Starting the Frontend

Since the frontend is built with plain HTML, CSS, and JavaScript, you can run it in two ways:

### 1. Open Directly in Browser
- Navigate to the frontend folder.
- Open `index.html` by double-clicking or right-click → **Open With → Browser**.

### 2. Serve via a Simple HTTP Server (Recommended)
If you face CORS issues or want a proper server environment:

```bash
# Navigate to the frontend folder
cd frontend

# Start a simple HTTP server on port 5500
python -m http.server 5500
````

* Open the following in your browser:

```
http://localhost:5500/index.html

```
---

## Using the App

1. **Select Dataset**

   * Choose `ILC` or `IN-ABS` from the dropdown.

2. **Select Entries**

   * Optionally enter a specific entry ID or number of entries to process.

3. **Upload File (Optional)**

   * Upload a legal document (JSON/text) to summarize.

4. **Run Pipeline**

   * Click **Run Pipeline**.
   * The app shows progress bars for each stage:

     * Cleaning
     * Chunking
     * Summarization
     * Evaluation

5. **View Results**

   * Once complete, view:

     * Original Legal Text
     * Reference Summary
     * AI-generated Summary
     * Global ROUGE scores

---

## Notes

* Works for both ILC and IN-ABS datasets.
* Ensure backend server is running before using the frontend.
* Large datasets may take longer; the app uses a progress bar to indicate status.

---

## Technology Stack

* **Backend:** FastAPI, Python, Hugging Face Transformers, PyTorch
* **Frontend:** HTML, CSS, JavaScript
* **Datasets:** ILC, IN-ABS
* **Summarization:** T5-based abstractive model
* **Evaluation:** ROUGE metrics


