const dropZone = document.getElementById("drop-zone");
const fileInput = document.getElementById("file-input");
const preview = document.getElementById("preview");
const uploadBtn = document.getElementById("upload-btn");
const resetBtn = document.getElementById("reset-btn");
const dzContent = document.querySelector(".dz-content");

const resultContainer = document.getElementById("result-container");
const classLabel = document.getElementById("class-label");
const confBar = document.getElementById("conf-bar");
const scoreText = document.getElementById("confidence-score");
const statusTag = document.getElementById("status-tag");
const resultMessage = document.getElementById("result-message");

const historyList = document.getElementById("history-list");
const historyEmpty = document.getElementById("history-empty");
const refreshHistoryBtn = document.getElementById("refresh-history-btn");
const historyDetailCard = document.getElementById("history-detail-card");
const historyDetailImage = document.getElementById("history-detail-image");
const historyDetailTime = document.getElementById("history-detail-time");
const historyDetailFile = document.getElementById("history-detail-file");
const historyDetailLabel = document.getElementById("history-detail-label");
const historyDetailConfidence = document.getElementById("history-detail-confidence");
const historyDetailStatus = document.getElementById("history-detail-status");
const historyDetailMessage = document.getElementById("history-detail-message");
const deleteHistoryBtn = document.getElementById("delete-history-btn");

const API_BASE = window.location.protocol === "file:" ? "http://127.0.0.1:8000" : "";
let selectedFile = null;
let selectedHistoryId = null;

function apiUrl(path) {
    return `${API_BASE}${path}`;
}

async function requestJson(path, options = {}) {
    const response = await fetch(apiUrl(path), options);
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
        throw new Error(data.detail || "Server response error.");
    }
    return data;
}

["dragenter", "dragover", "dragleave", "drop"].forEach((eventName) => {
    dropZone.addEventListener(eventName, (event) => event.preventDefault());
});

dropZone.addEventListener("dragover", () => dropZone.classList.add("active"));
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("active"));

dropZone.addEventListener("drop", (event) => {
    dropZone.classList.remove("active");
    const file = event.dataTransfer.files[0];
    if (file && file.type.startsWith("image/")) {
        handleFile(file);
    }
});

dropZone.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => {
    if (fileInput.files[0]) {
        handleFile(fileInput.files[0]);
    }
});

function handleFile(file) {
    selectedFile = file;
    const reader = new FileReader();
    reader.onload = (event) => {
        preview.src = event.target.result;
        preview.hidden = false;
        dzContent.style.display = "none";
        uploadBtn.disabled = false;
    };
    reader.readAsDataURL(file);
}

function formatStatus(status) {
    return (status || "unknown").replaceAll("_", " ");
}

function showResult(label, confidence, message, status) {
    resultContainer.hidden = false;
    classLabel.innerText = label;
    resultMessage.innerText = message || "Model prediction completed.";

    if (status === "invalid_input" || status === "unrecognized") {
        classLabel.style.color = "#dc2626";
        scoreText.innerText =
            typeof confidence === "number" ? `${(confidence * 100).toFixed(2)}%` : confidence;
        confBar.style.width = "0%";
        confBar.style.backgroundColor = "#ef4444";
        statusTag.innerText = "Manual review needed";
    } else {
        classLabel.style.color = "#059669";

        let displayConf = "";
        let widthVal = "0%";
        if (typeof confidence === "number") {
            displayConf = `${(confidence * 100).toFixed(2)}%`;
            widthVal = displayConf;
        } else {
            displayConf = confidence;
            widthVal = confidence;
        }

        scoreText.innerText = displayConf;
        confBar.style.width = widthVal;
        confBar.style.backgroundColor = "#10b981";
        statusTag.innerText = "Prediction complete";
    }
}

function renderHistoryList(items) {
    historyList.innerHTML = "";
    historyEmpty.hidden = items.length > 0;

    items.forEach((item) => {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "history-item";
        button.dataset.recordId = String(item.id);
        if (item.id === selectedHistoryId) {
            button.classList.add("active");
        }
        button.innerHTML = `
            <span class="history-item-top">
                <span class="history-item-label">${item.predicted_label}</span>
                <span class="history-item-time">${item.created_at}</span>
            </span>
            <span class="history-item-bottom">
                <span class="history-item-file">${item.original_filename}</span>
                <span class="history-item-confidence">${item.confidence_text}</span>
            </span>
        `;
        button.addEventListener("click", () => loadHistoryDetail(item.id));
        historyList.appendChild(button);
    });
}

function clearHistoryDetail() {
    selectedHistoryId = null;
    historyDetailCard.hidden = true;
    historyDetailImage.removeAttribute("src");
}

function fillHistoryDetail(record) {
    selectedHistoryId = record.id;
    historyDetailCard.hidden = false;
    historyDetailImage.src = apiUrl(record.image_url);
    historyDetailTime.innerText = record.created_at;
    historyDetailFile.innerText = record.original_filename;
    historyDetailLabel.innerText = record.predicted_label;
    historyDetailConfidence.innerText = record.confidence_text;
    historyDetailStatus.innerText = formatStatus(record.status);
    historyDetailMessage.innerText = record.message;
    deleteHistoryBtn.dataset.recordId = String(record.id);
}

async function loadHistoryDetail(recordId) {
    const record = await requestJson(`/history/${recordId}`);
    fillHistoryDetail(record);

    document.querySelectorAll(".history-item").forEach((itemButton) => {
        itemButton.classList.remove("active");
    });
    const matching = [...document.querySelectorAll(".history-item")].find(
        (itemButton) => Number(itemButton.dataset.recordId) === record.id
    );
    if (matching) {
        matching.classList.add("active");
    }
}

async function refreshHistory(preferredId = null) {
    try {
        const payload = await requestJson("/history");
        const items = payload.items || [];
        const recordToSelect =
            preferredId && items.some((item) => item.id === preferredId)
                ? preferredId
                : (items[0] && items[0].id) || null;

        renderHistoryList(items);
        if (recordToSelect !== null) {
            await loadHistoryDetail(recordToSelect);
        } else {
            clearHistoryDetail();
        }
    } catch (error) {
        console.error(error);
        historyEmpty.hidden = false;
        historyEmpty.innerText = `Unable to load history. ${error.message}`;
        historyList.innerHTML = "";
        clearHistoryDetail();
    }
}

uploadBtn.addEventListener("click", async () => {
    if (!selectedFile) {
        return;
    }

    const formData = new FormData();
    formData.append("file", selectedFile);

    uploadBtn.innerText = "Running inference...";
    uploadBtn.disabled = true;

    try {
        const data = await requestJson("/predict", {
            method: "POST",
            body: formData,
        });

        const finalLabel = data.prediction || data.class || "Unknown";
        const finalConf = data.confidence || "0%";
        const finalMessage = data.message || "";
        const finalStatus = data.status || "success";
        showResult(finalLabel, finalConf, finalMessage, finalStatus);
        await refreshHistory(data.history_id || null);
    } catch (error) {
        console.error(error);
        alert(`Unable to connect to the demo backend.\n${error.message}`);
    } finally {
        uploadBtn.innerText = "Start AI diagnosis";
        uploadBtn.disabled = false;
    }
});

refreshHistoryBtn.addEventListener("click", async () => {
    await refreshHistory(selectedHistoryId);
});

deleteHistoryBtn.addEventListener("click", async () => {
    const recordId = Number(deleteHistoryBtn.dataset.recordId);
    if (!recordId) {
        return;
    }

    const shouldDelete = window.confirm("Delete this history record?");
    if (!shouldDelete) {
        return;
    }

    try {
        await requestJson(`/history/${recordId}`, { method: "DELETE" });
        await refreshHistory();
    } catch (error) {
        console.error(error);
        alert(`Unable to delete history record.\n${error.message}`);
    }
});

resetBtn.addEventListener("click", () => {
    location.reload();
});

window.addEventListener("load", async () => {
    await refreshHistory();
});
