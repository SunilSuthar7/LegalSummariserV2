/* ================= GLOBAL SAFETY ================= */
// Prevent any accidental form submission reloads
document.addEventListener("submit", (e) => {
  e.preventDefault()
})

/* ================= DOM ELEMENTS ================= */
const runBtn = document.getElementById("runBtn")
const uploadFile = document.getElementById("uploadFile")

const datasetSelect = document.getElementById("dataset")
const numEntriesSelect = document.getElementById("numEntries")

const datasetSection = document.getElementById("datasetSection")
const uploadSection = document.getElementById("uploadSection")
const modeRadios = document.querySelectorAll('input[name="mode"]')

const cleaningBar = document.getElementById("cleaningBar")
const bertBar = document.getElementById("bertBar")
const t5Bar = document.getElementById("t5Bar")

const status1 = document.getElementById("status1")
const status2 = document.getElementById("status2")
const status3 = document.getElementById("status3")

const summaryText = document.getElementById("summaryText")
const resultCard = document.getElementById("resultCard")
const errorEl = document.getElementById("error")
const progressSection = document.getElementById("progressSection")

/* ================= STATE ================= */
const API = "http://127.0.0.1:8000"
let mode = "dataset"
let currentSessionId = null

/* ================= STAGE TRACKING ================= */
const stageConfig = {
  "Cleaning": { bar: cleaningBar, status: status1, order: 1 },
  "LegalBERT Extractive": { bar: bertBar, status: status2, order: 2 },
  "T5 Abstractive": { bar: t5Bar, status: status3, order: 3 }
}

let completedStages = new Set()
let animatingStages = new Set()
let stageProgress = {
  "Cleaning": 0,
  "LegalBERT Extractive": 0,
  "T5 Abstractive": 0
}

/* ================= HELPERS ================= */
function setBar(bar, percent, statusEl = null) {
  if (bar) {
    bar.style.width = percent + "%"
    if (statusEl) {
      if (percent === 0) {
        statusEl.textContent = "Pending"
        statusEl.style.color = "var(--neutral-600)"
      } else if (percent === 100) {
        statusEl.textContent = "✓ Complete"
        statusEl.style.color = "var(--success)"
      } else {
        statusEl.textContent = "In Progress"
        statusEl.style.color = "var(--warning)"
      }
    }
  }
}

function smoothAnimateBar(bar, targetPercent, statusEl = null, callback = null) {
  if (!bar) return
  
  const currentWidth = parseFloat(bar.style.width) || 0
  
  // Only animate if we're increasing the width
  if (currentWidth < targetPercent) {
    // Smooth increment that varies based on how far we need to go
    const remaining = targetPercent - currentWidth
    const increment = Math.max(remaining / 15, 1) // Slower, more gradual animation
    const newWidth = Math.min(currentWidth + increment, targetPercent)
    bar.style.width = newWidth + "%"
    
    if (newWidth < targetPercent) {
      setTimeout(() => smoothAnimateBar(bar, targetPercent, statusEl, callback), 200)
    } else {
      // Reached target, update status
      setBar(bar, targetPercent, statusEl)
      if (callback) callback()
    }
  }
}

function resetUI() {
  setBar(cleaningBar, 0, status1)
  setBar(bertBar, 0, status2)
  setBar(t5Bar, 0, status3)
  
  completedStages = new Set()
  animatingStages = new Set()
  stageProgress = {
    "Cleaning": 0,
    "LegalBERT Extractive": 0,
    "T5 Abstractive": 0
  }
  
  resultCard.style.display = "none"
  errorEl.textContent = ""
  progressSection.classList.remove("active")
  currentSessionId = null
}

function showProgress() {
  progressSection.classList.add("active")
}

function hideProgress() {
  progressSection.classList.remove("active")
}

/* ================= MODE TOGGLE ================= */
modeRadios.forEach(radio => {
  radio.addEventListener("change", (e) => {
    mode = e.target.value

    if (mode === "upload") {
      uploadSection.classList.add("active")
      datasetSection.classList.remove("active")
    } else {
      uploadSection.classList.remove("active")
      datasetSection.classList.add("active")
    }

    resetUI()
  })
})

/* ================= DATASET BUTTON SELECTION ================= */
document.querySelectorAll('.dataset-btn').forEach(btn => {
  btn.addEventListener('click', (e) => {
    e.preventDefault()
    e.stopPropagation()
    
    document.querySelectorAll('.dataset-btn').forEach(b => b.classList.remove('active'))
    btn.classList.add('active')
    
    const datasetValue = btn.getAttribute('data-dataset')
    document.getElementById('dataset').value = datasetValue
    
    console.log("Dataset selected:", datasetValue)
  })
})

/* ================= FILE UPLOAD HANDLING ================= */
const uploadArea = document.querySelector('.upload-area')
const uploadPlaceholder = document.querySelector('.upload-placeholder')
const fileInfo = document.getElementById('fileInfo')
const fileName = document.getElementById('fileName')
const fileRemove = document.getElementById('fileRemove')

if (uploadPlaceholder) {
  uploadPlaceholder.addEventListener('click', (e) => {
    e.preventDefault()
    e.stopPropagation()
    uploadFile.click()
  })
}

if (uploadArea) {
  uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault()
    e.stopPropagation()
    uploadArea.classList.add('drag-over')
  })

  uploadArea.addEventListener('dragleave', (e) => {
    e.preventDefault()
    e.stopPropagation()
    uploadArea.classList.remove('drag-over')
  })

  uploadArea.addEventListener('drop', (e) => {
    e.preventDefault()
    e.stopPropagation()
    uploadArea.classList.remove('drag-over')
    
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      uploadFile.files = e.dataTransfer.files
      updateFileInfo()
    }
  })
}

if (uploadFile) {
  uploadFile.addEventListener('change', updateFileInfo)
}

function updateFileInfo() {
  if (uploadFile && uploadFile.files && uploadFile.files.length > 0) {
    const file = uploadFile.files[0]
    fileName.textContent = `✓ ${file.name} (${(file.size / 1024).toFixed(2)} KB)`
    fileInfo.style.display = 'flex'
    if (uploadArea) {
      uploadArea.style.display = 'none'
    }
    console.log("File selected:", file.name)
  }
}

if (fileRemove) {
  fileRemove.addEventListener('click', (e) => {
    e.preventDefault()
    e.stopPropagation()
    
    uploadFile.value = ''
    fileInfo.style.display = 'none'
    if (uploadArea) {
      uploadArea.style.display = 'block'
    }
    console.log("File cleared")
  })
}

/* ================= RUN PIPELINE ================= */
if (runBtn) {
  runBtn.addEventListener("click", async (e) => {
    e.preventDefault()
    e.stopPropagation()
    e.stopImmediatePropagation()

    resetUI()
    showProgress()

    const fd = new FormData()
    fd.append("mode", mode)

    if (mode === "upload") {
      if (!uploadFile || !uploadFile.files || uploadFile.files.length === 0) {
        errorEl.textContent = "❌ Please upload a document first."
        hideProgress()
        return
      }
      fd.append("file", uploadFile.files[0])
      console.log("Uploading file:", uploadFile.files[0].name)
    }

    if (mode === "dataset") {
      const dataset = datasetSelect.value
      const n = numEntriesSelect.value

      if (!dataset || !n) {
        errorEl.textContent = "❌ Please select dataset and number of entries."
        hideProgress()
        return
      }

      fd.append("dataset", dataset)
      fd.append("n", n)
      console.log("Dataset mode:", dataset, "Entries:", n)
    }

    let res, data
    try {
      console.log("Sending request to backend...")
      res = await fetch(`${API}/run_pipeline`, {
        method: "POST",
        body: fd
      })
      data = await res.json()
      console.log("Backend response:", res.status, data)
    } catch (err) {
      console.error("Fetch error:", err)
      errorEl.textContent = "❌ Failed to contact backend. Is the server running on http://127.0.0.1:8000?"
      hideProgress()
      return
    }

    if (!res.ok) {
      console.error("API error:", data)
      errorEl.textContent = `❌ ${data.detail || "Pipeline failed to start."}`
      hideProgress()
      return
    }

    currentSessionId = data.session_id
    console.log("Pipeline started, session:", currentSessionId)
    console.log("Starting to poll status...")
    pollPipeline(currentSessionId)
  })
}

/* ================= PIPELINE POLLING ================= */
async function pollPipeline(sessionId) {
  let pollCount = 0
  const maxPolls = 240
  let lastWarningLog = 0
  
  const interval = setInterval(async () => {
    pollCount++
    
    if (pollCount > maxPolls) {
      clearInterval(interval)
      errorEl.textContent = "❌ Pipeline timeout. Request took too long."
      hideProgress()
      return
    }

    let res, status
    try {
      res = await fetch(
        `${API}/pipeline_status?session_id=${sessionId}`
      )
      
      if (res.status === 404) {
        if (pollCount - lastWarningLog >= 5) {
          console.warn(`[Poll ${pollCount}] Session not found yet, retrying...`)
          lastWarningLog = pollCount
        }
        return
      }
      
      if (!res.ok) {
        throw new Error(`HTTP ${res.status}`)
      }
      
      status = await res.json()
    } catch (err) {
      console.error("Poll error:", err)
      if (pollCount > 20) {
        clearInterval(interval)
        errorEl.textContent = "❌ Lost connection while polling."
        hideProgress()
      }
      return
    }

    if (status.error) {
      console.error("Pipeline error:", status.error)
      clearInterval(interval)
      errorEl.textContent = `❌ Pipeline Error: ${status.error}`
      hideProgress()
      return
    }

    const stages = Array.isArray(status.stages) ? status.stages : []
    
    console.log(`[Poll ${pollCount}] Completed stages:`, stages, "Pipeline completed:", status.completed)

    // Process each stage that just completed
    stages.forEach(stageName => {
      if (!completedStages.has(stageName)) {
        console.log(`✓ Stage JUST COMPLETED: ${stageName}`)
        completedStages.add(stageName)
        animatingStages.delete(stageName)
        
        const config = stageConfig[stageName]
        if (config) {
          // Animate to 100%
          smoothAnimateBar(config.bar, 100, config.status)
        }
      }
    })

    // For stages NOT completed yet, only animate if:
    // 1. Previous stage is already complete
    // 2. Current stage is in the stages list
    Object.keys(stageConfig).forEach(stageName => {
      const config = stageConfig[stageName]
      const isCompleted = completedStages.has(stageName)
      const isStarted = stages.includes(stageName)
      const previousStages = Object.keys(stageConfig)
        .filter(s => stageConfig[s].order < config.order)
      const allPreviousComplete = previousStages.every(s => completedStages.has(s))
      
      // Only animate if: not completed, has started, previous stage complete, and not already animating
      if (!isCompleted && isStarted && allPreviousComplete && !animatingStages.has(stageName) && !status.completed) {
        console.log(`→ Stage IN PROGRESS: ${stageName}`)
        animatingStages.add(stageName)
        
        // Animate to 85% while in progress
        smoothAnimateBar(config.bar, 85, config.status)
      }
    })

    if (status.completed) {
      clearInterval(interval)

      if (!status.results || !status.results.length) {
        console.error("No results returned")
        errorEl.textContent = "⚠️ Pipeline completed but no summary returned."
        hideProgress()
        return
      }

      // Ensure ALL stages show 100% complete
      console.log("✓ Pipeline COMPLETED - marking all stages as complete")
      Object.keys(stageConfig).forEach(stageName => {
        const config = stageConfig[stageName]
        setBar(config.bar, 100, config.status)
      })

      // ---------- DISPLAY RESULTS ----------
      let output = ""
      status.results.forEach((r, idx) => {
        output += `📄 Summary ${idx + 1}:\n`
        output += "─".repeat(60) + "\n"
        output += (r.summary_text || "No summary generated") + "\n\n"
      })

      summaryText.textContent = output.trim()
      resultCard.style.display = "block"
      console.log("Results displayed successfully")
      
      setTimeout(() => {
        resultCard.scrollIntoView({ behavior: 'smooth' })
      }, 300)
      
      hideProgress()
    }
  }, 1500)
}
