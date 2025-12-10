const runBtn = document.getElementById("runBtn");
const datasetEl = document.getElementById("dataset");
const numEntriesEl = document.getElementById("numEntries");
const entryIdEl = document.getElementById("entryId");
const fileInput = document.getElementById("fileInput");

const errorEl = document.getElementById("error");
const resultsEl = document.getElementById("results");
const avgRougeEl = document.getElementById("avgRouge");
const entriesContainer = document.getElementById("entriesContainer");

const fillCleaning = document.getElementById("fillCleaning");
const fillChunking = document.getElementById("fillChunking");
const fillSummarization = document.getElementById("fillSummarization");
const fillEvaluation = document.getElementById("fillEvaluation");

let pollInterval;

const API_BASE =
  window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1"
    ? "http://127.0.0.1:8000"
    : window.location.origin;

function resetProgress() {
  fillCleaning.style.width = "0%";
  fillChunking.style.width = "0%";
  fillSummarization.style.width = "0%";
  fillEvaluation.style.width = "0%";
}

function updateProgress(stages) {
  stages.forEach(stage => {
    const stageName = stage.stage.toLowerCase();
    const width = stage.status === "completed" ? "100%" : "50%";
    if (stageName.includes("clean")) fillCleaning.style.width = width;
    if (stageName.includes("chunk")) fillChunking.style.width = width;
    if (stageName.includes("summar")) fillSummarization.style.width = width;
    if (stageName.includes("evalu")) fillEvaluation.style.width = width;
  });
}

function setRunningState(isRunning) {
  runBtn.disabled = isRunning;
  runBtn.innerHTML = isRunning ? "Processing..." : "Run Pipeline";
  datasetEl.disabled = isRunning;
  numEntriesEl.disabled = isRunning;
  entryIdEl.disabled = isRunning;
  fileInput.disabled = isRunning;
}

async function fetchAndCombineSessionData(sessionId, dataset) {
  const datasetKey = dataset === "ILC" ? "ilc" : "in-abs";

  const [cleanedRes, t5Res, rougeRes] = await Promise.all([
    fetch(`${API_BASE}/sessions/${sessionId}/cleaned.json`),
    fetch(`${API_BASE}/sessions/${sessionId}/t5_${datasetKey}_final.json`),
    fetch(`${API_BASE}/sessions/${sessionId}/rouge_${datasetKey}.json`)
  ]);

  const cleaned = await cleanedRes.json();
  const t5 = await t5Res.json();
  const rouge = await rougeRes.json();

  // Map by ID
  const cleanedMap = {};
  cleaned.forEach(e => cleanedMap[e.id] = e);
  const t5Map = {};
  t5.forEach(e => t5Map[e.id] = e);

  // Combine data
  const combined = Object.keys(cleanedMap).map(id => ({
    id,
    input_text: cleanedMap[id].input_text ?? "N/A",
    reference_summary: cleanedMap[id].summary_text ?? "N/A",
    refined_summary_improved: t5Map[id]?.refined_summary_improved ?? "N/A"
  }));

  console.log("COMBINED DATA:", combined); 

  return { combined, rougeScores: rouge.scores || {} };
}

runBtn.onclick = async () => {
  errorEl.style.display = "none";
  resultsEl.style.display = "block";
  // Do NOT clear resultsEl.innerHTML, just clear entriesContainer and avgRouge
  entriesContainer.style.display = "block";
  entriesContainer.innerHTML = "";
  avgRougeEl.innerHTML = "";
  resetProgress();
  setRunningState(true);

  const formData = new FormData();
  formData.append("dataset", datasetEl.value);

  const specificEntryId = entryIdEl.value.trim();
  if (specificEntryId && Number.parseInt(specificEntryId) > 0) {
    formData.append("n", Number.parseInt(specificEntryId));
    formData.append("entry_id", Number.parseInt(specificEntryId));
  } else {
    formData.append("n", Number(numEntriesEl.value));
  }

  if (fileInput.files.length > 0) formData.append("file", fileInput.files[0]);

  try {
    const res = await fetch(`${API_BASE}/run_pipeline`, { method: "POST", body: formData });
    const data = await res.json();
    if (!res.ok) throw new Error(data?.error || res.statusText);

    const sessionId = data.session_id;

    pollInterval = setInterval(async () => {
      try {
        const statusRes = await fetch(`${API_BASE}/pipeline_status?session_id=${sessionId}`);
        const statusData = await statusRes.json();

        if (statusData.stages) updateProgress(statusData.stages);

        if (statusData.completed) {
          clearInterval(pollInterval);
          setRunningState(false);

          const { combined, rougeScores } = await fetchAndCombineSessionData(sessionId, datasetEl.value);

          avgRougeEl.innerHTML = `
            <strong>Global ROUGE Scores:</strong>
            ROUGE-1: ${rougeScores.rouge1 !== undefined ? (rougeScores.rouge1 * 100).toFixed(1) + '%' : "N/A"} |
            ROUGE-2: ${rougeScores.rouge2 !== undefined ? (rougeScores.rouge2 * 100).toFixed(1) + '%' : "N/A"} |
            ROUGE-L: ${rougeScores.rougeL !== undefined ? (rougeScores.rougeL * 100).toFixed(1) + '%' : "N/A"}
          `;

          // Use DocumentFragment for better performance & proper rendering
          const fragment = document.createDocumentFragment();
          combined.forEach((entry, idx) => {
            const card = document.createElement("div");
            card.className = "entry-card";

            card.innerHTML = `
              <div class="entry-header">Document ${idx + 1}</div>
              <div class="entry-content">
                <div class="text-section">
                  <strong>Original Legal Text:</strong>
                  <div class="text-content original">${entry.input_text}</div>
                </div>
                <div class="text-section">
                  <strong>Reference Summary:</strong>
                  <div class="text-content reference">${entry.reference_summary}</div>
                </div>
                <div class="text-section">
                  <strong>AI Generated Summary:</strong>
                  <div class="text-content generated">${entry.refined_summary_improved}</div>
                </div>
              </div>
            `;
            fragment.appendChild(card);
          });

          entriesContainer.innerHTML = "";
          entriesContainer.appendChild(fragment);
          entriesContainer.style.display = "flex";
          entriesContainer.style.visibility = "visible";
          entriesContainer.style.opacity = "1";
          entriesContainer.style.minHeight = "200px";
          console.log("Appended fragment to entriesContainer", entriesContainer, fragment);
          console.log("Computed style:", window.getComputedStyle(entriesContainer));
          console.log("Parent element:", entriesContainer.parentElement);
          console.log("Parent computed style:", window.getComputedStyle(entriesContainer.parentElement));
        }
      } catch (e) {
        clearInterval(pollInterval);
        setRunningState(false);
        errorEl.style.display = "block";
        errorEl.textContent = "Error fetching pipeline status: " + e.message;
      }
    }, 1000);

  } catch (err) {
    setRunningState(false);
    errorEl.style.display = "block";
    errorEl.textContent = "Error: " + err.message;
  }
};

window.addEventListener("beforeunload", () => {
  if (pollInterval) clearInterval(pollInterval);
});
