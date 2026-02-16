// ============================================================================
// ML PLAYGROUND - CORE WORKFLOW ONLY
// ============================================================================

const API_URL = "http://localhost:5000";

// State
const state = {
    file: null,
    csvData: null,
    columns: [],
    selectedFeatures: [],
    selectedTarget: "",
    selectedAlgorithm: "",
    selectedMetric: "accuracy" // default
};

// DOM Elements
const fileInput = document.getElementById("file-input");
const fileName = document.getElementById("file-name");
const previewBlock = document.getElementById("preview-block");
const featuresBlock = document.getElementById("features-block");
const targetBlock = document.getElementById("target-block");
const algorithmBlock = document.getElementById("algorithm-block");
const trainBlock = document.getElementById("train-block");
const resultBlock = document.getElementById("results-block");

const rowCount = document.getElementById("row-count");
const colCount = document.getElementById("col-count");
const tableHead = document.getElementById("table-head");
const tableBody = document.getElementById("table-body");
const featuresList = document.getElementById("features-list");
const targetSelect = document.getElementById("target-select");
const algorithmSelect = document.getElementById("algorithm-select");
const trainButton = document.getElementById("train-button");
const statusMessage = document.getElementById("status-message");
const metricDisplay = document.getElementById("metric-display");
const metricTab = document.getElementById("metrics-tab");
const metricList = document.getElementById("metrics-list");

// ============================================================================
// EVENT LISTENERS
// ============================================================================

fileInput.addEventListener("change", handleFileUpload);
algorithmSelect.addEventListener("change", (e) => {
    state.selectedAlgorithm = e.target.value;
});
if (metricList) {
    metricList.addEventListener("change", (e) => {
        if (e.target && e.target.name === "metric-radio") {
            state.selectedMetric = e.target.value;
        }
    });
}
trainButton.addEventListener("click", handleTrain);

// ============================================================================
// FILE UPLOAD & CSV PARSING
// ============================================================================

async function handleFileUpload(event) {
    const file = event.target.files[0];
    
    if (!file || !file.name.endsWith(".csv")) {
        fileName.textContent = "❌ Please select a CSV file";
        fileName.style.color = "var(--error-color)";
        return;
    }
    
    state.file = file;
    fileName.textContent = `✓ ${file.name}`;
    fileName.style.color = "var(--success-color)";
    
    // Read and parse CSV
    const reader = new FileReader();
    reader.onload = (e) => {
        const csvText = e.target.result;
        parseCSV(csvText);
    };
    reader.readAsText(file);
}

function parseCSV(csvText) {
    const lines = csvText.trim().split("\n");
    const headers = lines[0].split(",").map(h => h.trim());
    
    state.columns = headers;
    state.csvData = lines;
    
    // Show blocks
    previewBlock.style.display = "block";
    featuresBlock.style.display = "block";
    targetBlock.style.display = "block";
    algorithmBlock.style.display = "block";
    trainBlock.style.display = "block";
    
    // Display preview
    displayPreview(headers, lines);
    
    // Populate feature list
    populateFeatures(headers);
    
    // Populate target select
    populateTargetSelect(headers);
}

function displayPreview(headers, lines) {
    rowCount.textContent = lines.length - 1; // -1 for header
    colCount.textContent = headers.length;
    
    // Table header
    tableHead.innerHTML = headers.map(h => `<th>${h}</th>`).join("");
    
    // Table body (first 10 rows)
    tableBody.innerHTML = "";
    for (let i = 1; i < Math.min(11, lines.length); i++) {
        const cells = lines[i].split(",").map(c => c.trim());
        const row = cells.map(c => `<td>${c}</td>`).join("");
        tableBody.innerHTML += `<tr>${row}</tr>`;
    }
}

function populateFeatures(headers) {
    featuresList.innerHTML = headers.map(col => `
        <label class="checkbox">
            <input type="checkbox" class="feature-checkbox" data-col="${col}" checked>
            <span>${col}</span>
        </label>
    `).join("");
    
    // Update selected features
    document.querySelectorAll(".feature-checkbox").forEach(cb => {
        cb.addEventListener("change", updateSelectedFeatures);
    });
    
    state.selectedFeatures = headers; // All columns selected by default
}

function populateTargetSelect(headers) {
    targetSelect.innerHTML = '<option value="">-- Select Target Column --</option>';
    targetSelect.innerHTML += headers.map(col => `
        <option value="${col}">${col}</option>
    `).join("");
    
    targetSelect.addEventListener("change", (e) => {
        state.selectedTarget = e.target.value;
    });
}

function updateSelectedFeatures() {
    state.selectedFeatures = Array.from(document.querySelectorAll(".feature-checkbox:checked"))
        .map(cb => cb.dataset.col);
}

// ============================================================================
// DEMO DATASETS
// ============================================================================

async function loadDemoDataset(name) {
    // Map frontend button names to backend demo-dataset names
    const nameMap = {
        iris: "iris",
        titanic: "titanic",
        house_prices: "housing"
    };
    const endpoint = `/demo-dataset/${nameMap[name]}`;
    try {
        fileName.textContent = "🔄 Loading dataset...";
        const response = await fetch(endpoint);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const csvText = await response.text();
        state.file = new File([csvText], `${name}.csv`, { type: "text/csv" });
        fileName.textContent = `✓ ${name}.csv`;
        fileName.style.color = "var(--success-color)";
        parseCSV(csvText);
    } catch (error) {
        fileName.textContent = `❌ Error loading ${name}: ${error.message}`;
        fileName.style.color = "var(--error-color)";
    }
}

// ============================================================================
// TRAINING
// ============================================================================

async function handleTrain() {
    // Validation
    if (!state.file) {
        showStatus("❌ Please upload a CSV file", "error");
        return;
    }
    
    if (state.selectedFeatures.length === 0) {
        showStatus("❌ Please select at least one feature", "error");
        return;
    }
    
    if (!state.selectedTarget) {
        showStatus("❌ Please select a target column", "error");
        return;
    }
    
    if (!state.selectedAlgorithm) {
        showStatus("❌ Please select an algorithm", "error");
        return;
    }
    
    // Disable button during training
    trainButton.disabled = true;
    showStatus("⏳ Training model...", "loading");
    
    try {
        // Prepare form data
        const formData = new FormData();
        formData.append("file", state.file);
        formData.append("features", JSON.stringify(state.selectedFeatures));
        formData.append("target", state.selectedTarget);
        formData.append("algorithm", state.selectedAlgorithm);
        formData.append("metric", state.selectedMetric);
        
        // Log payload
        console.log("📤 TRAINING PAYLOAD:");
        console.log("  Features:", state.selectedFeatures);
        console.log("  Target:", state.selectedTarget);
        console.log("  Algorithm:", state.selectedAlgorithm);
        console.log("  File:", state.file.name);
        
        // Send request
        const response = await fetch(`${API_URL}/train`, {
            method: "POST",
            body: formData
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || "Training failed");
        }
        
        const result = await response.json();
        
        console.log("📥 TRAINING RESULT:", result);
        
        // Show results
        if (result.metric_name && result.metric_value !== undefined) {
            displayResults(result.metric_name, result.metric_value, result.plots || [], result.explanation);
            showStatus("✅ Training completed!", "success");
        } else if (result.error) {
            showStatus(`❌ ${result.error}`, "error");
        } else {
            throw new Error("Invalid response format");
        }
        
    } catch (error) {
        console.error("❌ ERROR:", error);
        showStatus(`❌ Error: ${error.message}`, "error");
    } finally {
        trainButton.disabled = false;
    }
}

function displayResults(metricName, metricValue, plots = [], explanation = "") {
    resultBlock.style.display = "block";
    metricDisplay.innerHTML = `<div style='font-size:1.5rem;font-weight:bold;color:var(--success-color);'>${metricName.toUpperCase()}: ${metricValue}</div>`;
    if (explanation) {
        metricDisplay.innerHTML += `<div style='margin-top:8px;font-size:1rem;color:var(--text-secondary);'>${explanation}</div>`;
    }
    // Show plots with explanations if present
    if (plots.length > 0) {
        metricDisplay.innerHTML += `<div style='margin-top:18px;'><strong>Generated Plots:</strong></div>`;
        plots.forEach(plotObj => {
            const url = `/ml-plot?path=${encodeURIComponent(plotObj.path)}`;
            metricDisplay.innerHTML += `
                <div style='margin:10px 0;'>
                    <img src='${url}' alt='ML Plot' style='max-width:100%;border:1px solid #e5e7eb;border-radius:6px;box-shadow:0 2px 8px #0001;'>
                    <div style='margin-top:4px;font-size:0.98rem;color:var(--text-secondary);'>${plotObj.explanation || ""}</div>
                </div>
            `;
        });
    }
}

function showStatus(message, type = "info") {
    statusMessage.textContent = message;
    statusMessage.className = `status-message ${type}`;
}

// ============================================================================
// INIT
// ============================================================================

document.addEventListener("DOMContentLoaded", () => {
    console.log("✅ ML Playground initialized");
    showStatus("Ready! Upload a CSV to get started.", "info");
});
