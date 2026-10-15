/* =========================================================
   AI ANSWER EVALUATOR — CONNECTED FRONTEND
   Reuses processed textbooks from the FastAPI/PostgreSQL library.
========================================================= */

const API_BASE = "http://127.0.0.1:8000";

let answerKeyFile = null;
let textbookId = null;
let textbookFilename = "";
let savedTextbooks = [];
let currentStudent = { name: "", usn: "", answerScript: null };
let currentEvaluation = null;
let marksChart = null;
let progressInterval = null;
let extractedOcrData = null;

document.addEventListener("DOMContentLoaded", () => {
    document.getElementById("answerKey").addEventListener("change", handleAnswerKeyChange);
    document.getElementById("newTextbookFile").addEventListener("change", handleNewTextbookChange);
    document.getElementById("answerScript").addEventListener("change", handleStudentFile);
    document.getElementById("textbookSelect").addEventListener("change", handleTextbookSelection);

    document.getElementById("loginForm").addEventListener("submit", event => {
        event.preventDefault();
        const username = document.getElementById("username").value.trim();
        const password = document.getElementById("password").value;

        // Demo-only frontend login, matching the original project behavior.
        if (username === "teacher" && password === "teacher123") {
            showPage("referencePage");
            loadSavedTextbooks();
        } else {
            alert("Invalid username or password.\n\nDemo login:\nUsername: teacher\nPassword: teacher123");
        }
    });
});

function showPage(pageId) {
    document.querySelectorAll(".page").forEach(page => page.classList.remove("active"));
    const page = document.getElementById(pageId);
    if (page) page.classList.add("active");
    window.scrollTo({ top: 0, behavior: "smooth" });
}

function handleAnswerKeyChange() {
    const input = document.getElementById("answerKey");
    answerKeyFile = input.files && input.files.length ? input.files[0] : null;
    document.getElementById("answerKeyName").textContent =
        answerKeyFile ? "✓ " + answerKeyFile.name : "No answer key selected";
    document.getElementById("selectedAnswerKeySummary").textContent =
        answerKeyFile ? answerKeyFile.name : "Upload an answer key PDF";
    updateReferenceStatus();
}

function handleNewTextbookChange() {
    const input = document.getElementById("newTextbookFile");
    const file = input.files && input.files.length ? input.files[0] : null;
    document.getElementById("newTextbookFileName").textContent =
        file ? "Selected: " + file.name : "No new textbook selected";
    document.getElementById("textbookDropZone").classList.toggle("has-file", Boolean(file));
}

function handleTextbookSelection() {
    const select = document.getElementById("textbookSelect");
    textbookId = select.value ? Number(select.value) : null;
    const textbook = savedTextbooks.find(item => Number(item.id) === textbookId);
    textbookFilename = textbook ? textbook.filename : "";

    document.getElementById("selectedTextbookSummary").textContent =
        textbook ? `${textbook.filename} · ${textbook.chunk_count} chunks` : "Select a textbook from your library";
    document.getElementById("studentTextbookSummary").textContent =
        textbook ? `${textbook.filename} is selected and will be reused.` : "Select a textbook in Reference Materials first.";

    renderTextbookLibrary();
    updateReferenceStatus();
}

async function loadSavedTextbooks(showMessage = false) {
    const select = document.getElementById("textbookSelect");
    const list = document.getElementById("textbookLibraryList");
    select.innerHTML = '<option value="">Loading saved textbooks...</option>';
    list.innerHTML = '<div class="library-empty"><span class="empty-icon">◌</span><strong>Loading your library</strong><p>Fetching processed textbooks...</p></div>';

    try {
        const response = await fetch(`${API_BASE}/api/textbooks`);
        const data = await readJsonResponse(response);
        if (!response.ok || data.success === false) {
            throw new Error(data.detail || "Could not load saved textbooks.");
        }

        const raw = Array.isArray(data) ? data : (data.textbooks || data.items || []);
        savedTextbooks = raw.map(item => ({
            id: Number(item.id ?? item.textbook_id),
            filename: item.filename || item.name || `Textbook ${item.id}`,
            chunk_count: Number(item.chunk_count ?? item.chunks_count ?? item.total_chunks ?? 0)
        })).filter(item => Number.isFinite(item.id) && item.id > 0 && item.chunk_count > 0);

        select.innerHTML = '<option value="">— Select a saved textbook —</option>';
        savedTextbooks.forEach(item => {
            const option = document.createElement("option");
            option.value = String(item.id);
            option.textContent = `${item.filename} (${item.chunk_count} chunks)`;
            select.appendChild(option);
        });

        // Keep current selection after refreshing, if it still exists.
        if (textbookId && savedTextbooks.some(item => item.id === textbookId)) {
            select.value = String(textbookId);
        } else {
            textbookId = null;
            textbookFilename = "";
            select.value = "";
        }

        document.getElementById("libraryCount").textContent =
            `${savedTextbooks.length} saved`;
        handleTextbookSelection();

        if (showMessage) {
            setProcessStatus(
                savedTextbooks.length ? "Textbook library refreshed." : "No processed textbooks yet. Add one below.",
                savedTextbooks.length ? "success" : "info"
            );
        }
    } catch (error) {
        console.error("Loading textbooks failed:", error);
        savedTextbooks = [];
        textbookId = null;
        textbookFilename = "";
        select.innerHTML = '<option value="">Could not load textbooks</option>';
        document.getElementById("libraryCount").textContent = "Unavailable";
        list.innerHTML = `<div class="library-empty error-empty"><span class="empty-icon">!</span><strong>Library unavailable</strong><p>${escapeHtml(error.message)}<br>Check that FastAPI is running.</p></div>`;
        document.getElementById("selectedTextbookSummary").textContent = "Textbook library unavailable";
        updateReferenceStatus();
    }
}

function renderTextbookLibrary() {
    const list = document.getElementById("textbookLibraryList");
    if (!savedTextbooks.length) {
        list.innerHTML = '<div class="library-empty"><span class="empty-icon">▤</span><strong>Your library is ready for its first textbook</strong><p>Upload a PDF below to process and save it here.</p></div>';
        return;
    }

    list.innerHTML = "";
    savedTextbooks.forEach(item => {
        const selected = item.id === textbookId;
        const row = document.createElement("button");
        row.type = "button";
        row.className = "library-item" + (selected ? " selected" : "");
        row.innerHTML = `
            <span class="book-file-icon">PDF</span>
            <span class="book-details"><strong>${escapeHtml(item.filename)}</strong><small>${item.chunk_count} searchable text chunks</small></span>
            <span class="book-selected">${selected ? "✓" : "›"}</span>`;
        row.addEventListener("click", () => {
            document.getElementById("textbookSelect").value = String(item.id);
            handleTextbookSelection();
        });
        list.appendChild(row);
    });
}

async function processNewTextbook() {
    const input = document.getElementById("newTextbookFile");
    const file = input.files && input.files.length ? input.files[0] : null;
    const button = document.getElementById("processTextbookBtn");

    if (!file) {
        alert("Choose a textbook PDF first.");
        return;
    }
    if (!file.name.toLowerCase().endsWith(".pdf") && file.type !== "application/pdf") {
        alert("Please choose a PDF file.");
        return;
    }

    const original = button.innerHTML;
    button.disabled = true;
    button.innerHTML = '<span class="button-spinner"></span> Processing textbook...';
    setProcessStatus("Uploading PDF, extracting text, and creating searchable chunks. This may take a little while.", "info");

    try {
        const formData = new FormData();
        formData.append("file", file);

        const response = await fetch(`${API_BASE}/api/textbooks/process`, {
            method: "POST",
            body: formData
        });
        const data = await readJsonResponse(response);

        if (!response.ok || data.success === false) {
            throw new Error(data.detail || data.message || "Textbook processing failed.");
        }

        setProcessStatus(data.message || "Textbook processed successfully. Refreshing your library...", "success");
        input.value = "";
        document.getElementById("newTextbookFileName").textContent = "No new textbook selected";
        document.getElementById("textbookDropZone").classList.remove("has-file");

        await loadSavedTextbooks();

        // Auto-select the newly processed textbook by filename when possible.
        const newTextbook = savedTextbooks.find(item => item.filename === file.name);
        if (newTextbook) {
            document.getElementById("textbookSelect").value = String(newTextbook.id);
            handleTextbookSelection();
        }

        if (textbookId) {
            setProcessStatus(`"${file.name}" is saved and selected. You can reuse it for multiple students.`, "success");
        } else {
            setProcessStatus("Processing finished. Select the textbook from your library to use it.", "success");
        }
    } catch (error) {
        console.error("Textbook processing failed:", error);
        setProcessStatus(error.message, "error");
        alert("Could not process the textbook.\n\n" + error.message);
    } finally {
        button.disabled = false;
        button.innerHTML = original;
    }
}

function setProcessStatus(message, type = "info") {
    const status = document.getElementById("textbookProcessStatus");
    status.textContent = message;
    status.className = `inline-status ${type}`;
}

function updateReferenceStatus() {
    const ready = Boolean(answerKeyFile && textbookId);
    document.getElementById("referenceStatus").classList.toggle("hidden", !ready);
    document.getElementById("selectedReferenceSummary").classList.toggle("references-ready", ready);
}

function goToStudentPage() {
    if (!textbookId) {
        alert("Please select a processed textbook from your library, or upload and process one first.");
        return;
    }
    if (!answerKeyFile) {
        alert("Please upload the answer key PDF first.");
        return;
    }
    document.getElementById("studentTextbookSummary").textContent =
        `${textbookFilename} is selected and will be reused.`;
    showPage("studentPage");
}

function handleStudentFile() {
    const input = document.getElementById("answerScript");
    const file = input.files && input.files.length ? input.files[0] : null;
    currentStudent.answerScript = file;
    document.getElementById("answerScriptName").textContent =
        file ? "✓ " + file.name : "No answer script selected";

    // Reset previous OCR state if a new file is chosen
    extractedOcrData = null;
    const previewContainer = document.getElementById("ocrPreviewContainer");
    if (previewContainer) previewContainer.classList.add("hidden");
    const ocrStatus = document.getElementById("ocrStatus");
    if (ocrStatus) {
        ocrStatus.textContent = "";
        ocrStatus.className = "inline-status";
    }
}

async function extractHandwriting() {
    const input = document.getElementById("answerScript");
    const file = input.files && input.files.length ? input.files[0] : null;
    if (!file) {
        alert("Please choose a student answer script PDF first.");
        return;
    }

    const studentNameInput = document.getElementById("studentName");
    const studentUsnInput = document.getElementById("studentUSN");
    const btn = document.getElementById("extractOcrBtn");
    const statusDiv = document.getElementById("ocrStatus");
    const previewContainer = document.getElementById("ocrPreviewContainer");
    const questionsList = document.getElementById("ocrQuestionsList");
    const summaryText = document.getElementById("ocrSummaryText");
    const cacheBadge = document.getElementById("ocrCacheBadge");
    const confidenceBadge = document.getElementById("ocrConfidenceBadge");

    btn.disabled = true;
    const originalBtnText = btn.innerHTML;
    btn.innerHTML = '<span class="button-spinner"></span> Extracting handwriting...';
    statusDiv.className = "inline-status running";
    statusDiv.textContent = "Analyzing handwriting strokes and detecting questions with OCR...";

    try {
        const formData = new FormData();
        formData.append("answer_script", file);
        if (studentNameInput.value.trim()) {
            formData.append("student_name", studentNameInput.value.trim());
        }
        if (studentUsnInput.value.trim()) {
            formData.append("student_usn", studentUsnInput.value.trim());
        }

        const response = await fetch(`${API_BASE}/api/ocr/extract`, {
            method: "POST",
            body: formData
        });
        const data = await readJsonResponse(response);

        if (!response.ok || data.success === false) {
            throw new Error(data.detail || "Handwriting extraction failed.");
        }

        extractedOcrData = data;

        // Auto-fill student name and USN if detected by OCR and fields are empty
        if (!studentNameInput.value.trim() && data.studentName) {
            studentNameInput.value = data.studentName;
        }
        if (!studentUsnInput.value.trim() && data.studentUSN) {
            studentUsnInput.value = data.studentUSN;
        }

        // Display results
        previewContainer.classList.remove("hidden");
        const confPercent = Math.round((data.confidence || 0.85) * 100);
        confidenceBadge.textContent = `${confPercent}% Confidence`;
        confidenceBadge.className = `confidence-badge ${confPercent >= 80 ? 'high' : 'medium'}`;

        if (data.cached) {
            cacheBadge.classList.remove("hidden");
            cacheBadge.textContent = "⚡ Loaded from Database (Instant)";
            statusDiv.className = "inline-status success";
            statusDiv.textContent = "✓ Found existing extraction in database. Zero wait time!";
        } else {
            cacheBadge.classList.add("hidden");
            statusDiv.className = "inline-status success";
            statusDiv.textContent = "✓ Handwriting extracted and saved to database successfully.";
        }

        const qMap = data.questions || {};
        const qNums = Object.keys(qMap).sort((a, b) => Number(a) - Number(b));
        summaryText.textContent = `${qNums.length} question(s) extracted from answer booklet`;

        // Render questions
        questionsList.innerHTML = "";
        if (qNums.length === 0) {
            questionsList.innerHTML = '<div class="ocr-question-empty">No clear question markers (Q12, Q13...) detected. Full text saved.</div>';
        } else {
            qNums.forEach(num => {
                const item = qMap[num];
                const card = document.createElement("div");
                card.className = "ocr-question-card";
                card.innerHTML = `
                    <div class="ocr-q-header">
                        <span class="ocr-q-num">Q${escapeHtml(String(num))}</span>
                        <span class="ocr-q-title">${escapeHtml(item.question || "(Question text in booklet)")}</span>
                    </div>
                    <div class="ocr-q-answer">${escapeHtml(item.answer || "")}</div>
                `;
                questionsList.appendChild(card);
            });
        }
    } catch (err) {
        console.error("OCR Extraction error:", err);
        statusDiv.className = "inline-status error";
        statusDiv.textContent = "Extraction failed: " + err.message;
        alert("OCR extraction failed:\n" + err.message);
    } finally {
        btn.disabled = false;
        btn.innerHTML = originalBtnText;
    }
}

function showEvaluationProgress() {
    const messages = [
        { title: "Preparing your answer script...", message: "Getting your uploaded documents ready." },
        { title: "Reading handwritten answers...", message: "Matching extracted answers with answer key." },
        { title: "Matching answers...", message: "Comparing detected answers with the answer key." },
        { title: "Evaluating with AI...", message: "Assessing answer relevance and generating marks and feedback." },
        { title: "Almost done...", message: "Finalizing the evaluation report." }
    ];
    const overlay = document.getElementById("evaluationLoading");
    const title = document.getElementById("loadingTitle");
    const message = document.getElementById("loadingMessage");
    const progressBar = document.getElementById("progressBar");
    const progressPercent = document.getElementById("progressPercent");
    let index = 0;
    let progress = 5;

    overlay.style.display = "flex";
    title.textContent = messages[0].title;
    message.textContent = messages[0].message;
    progressBar.style.width = `${progress}%`;
    progressPercent.textContent = "Evaluation in progress";

    progressInterval = setInterval(() => {
        progress = Math.min(progress + Math.random() * 4, 90);
        progressBar.style.width = `${progress}%`;
        index = Math.min(index + 1, messages.length - 1);
        title.textContent = messages[index].title;
        message.textContent = messages[index].message;
    }, 5000);

    return {
        stop() {
            if (progressInterval) clearInterval(progressInterval);
            progressInterval = null;
        },
        async complete() {
            if (progressInterval) clearInterval(progressInterval);
            progressInterval = null;
            progressBar.style.width = "100%";
            progressPercent.textContent = "Evaluation completed";
            title.textContent = "Evaluation complete";
            message.textContent = "Your evaluation report is ready.";
            await new Promise(resolve => setTimeout(resolve, 650));
        }
    };
}

async function evaluateStudent() {
    const name = document.getElementById("studentName").value.trim();
    const usn = document.getElementById("studentUSN").value.trim();
    const answerScript = document.getElementById("answerScript").files[0];

    if (!name || !usn) {
        alert("Please enter the student's name and USN.");
        return;
    }
    if (!answerKeyFile) {
        alert("Please upload the answer key on the Reference Materials page.");
        return;
    }
    if (!textbookId) {
        alert("Please select a processed textbook on the Reference Materials page.");
        return;
    }
    if (!answerScript && !extractedOcrData) {
        alert("Please upload the student's answer script PDF.");
        return;
    }

    currentStudent = { name, usn, answerScript };
    const button = document.querySelector(".evaluate-btn");
    const originalText = button.innerHTML;
    let progress = null;
    button.disabled = true;
    button.innerHTML = '<span class="button-spinner"></span> Evaluating...';

    try {
        progress = showEvaluationProgress();
        const formData = new FormData();
        formData.append("student_name", name);
        formData.append("student_usn", usn);
        formData.append("textbook_id", String(textbookId));
        formData.append("answer_key", answerKeyFile);

        // Reuse pre-extracted OCR data to skip re-running OCR during evaluation!
        if (extractedOcrData && extractedOcrData.scriptHash) {
            formData.append("script_hash", extractedOcrData.scriptHash);
            if (extractedOcrData.questions) {
                formData.append("extracted_answers", JSON.stringify(extractedOcrData.questions));
            }
        }
        if (answerScript) {
            formData.append("answer_script", answerScript);
        }

        const response = await fetch(`${API_BASE}/api/evaluate`, {
            method: "POST",
            body: formData
        });
        const data = await readJsonResponse(response);

        if (!response.ok) {
            throw new Error(data.detail || "Evaluation failed. Please try again.");
        }

        currentEvaluation = {
            id: data.evaluationId,
            obtained: Number(data.obtainedMarks || 0),
            total: Number(data.totalMarks || 0),
            percentage: Number(data.percentage || 0),
            textbookFilename: data.textbookFilename || textbookFilename,
            questions: (data.questions || []).map(q => ({
                question: `Q${q.questionNumber}`,
                questionText: q.question || "",
                max: Number(q.maxMarks || 0),
                obtained: Number(q.marks || 0),
                evaluation: q.feedback || q.justification || "No feedback available",
                studentAnswer: q.studentAnswer || "",
                modelAnswer: q.modelAnswer || "",
                correctConcepts: q.correctConcepts || [],
                missingConcepts: q.missingConcepts || [],
                incorrectConcepts: q.incorrectConcepts || [],
                textbookSupport: q.textbookSupport || "",
                ocrInterpretation: q.ocrInterpretation || ""
            }))
        };

        await progress.complete();
        displayResult();
    } catch (error) {
        console.error("Evaluation error:", error);
        alert("Could not evaluate the answer script.\n\n" + error.message + "\n\nCheck that the backend is running and see its terminal for details.");
    } finally {
        if (progress) progress.stop();
        document.getElementById("evaluationLoading").style.display = "none";
        button.disabled = false;
        button.innerHTML = originalText;
    }
}

async function readJsonResponse(response) {
    const text = await response.text();
    try {
        return text ? JSON.parse(text) : {};
    } catch {
        throw new Error("The backend returned an invalid response. Check the FastAPI terminal.");
    }
}

function displayResult() {
    const result = currentEvaluation;
    document.getElementById("resultStudentName").textContent = currentStudent.name;
    document.getElementById("resultUSN").textContent = `USN: ${currentStudent.usn}`;
    document.getElementById("obtainedMarks").textContent = formatMarks(result.obtained);
    document.getElementById("totalMarks").textContent = formatMarks(result.total);
    document.getElementById("percentage").textContent = `${result.percentage.toFixed(1)}%`;
    document.getElementById("rating").textContent = getRating(result.percentage);
    document.getElementById("resultTextbook").textContent =
        result.textbookFilename ? `Reference: ${result.textbookFilename}` : "";
    createChart(result.obtained, result.total);
    showPage("resultPage");
}

function formatMarks(value) {
    return Number.isInteger(value) ? String(value) : value.toFixed(2).replace(/0+$/, "").replace(/\.$/, "");
}

function getRating(percentage) {
    if (percentage >= 90) return "Excellent performance";
    if (percentage >= 75) return "Very good performance";
    if (percentage >= 60) return "Good performance";
    if (percentage >= 40) return "Developing";
    return "Needs improvement";
}

function createChart(obtained, total) {
    const remaining = Math.max(0, total - obtained);
    const context = document.getElementById("marksChart").getContext("2d");
    if (marksChart) marksChart.destroy();
    marksChart = new Chart(context, {
        type: "doughnut",
        data: {
            labels: ["Marks obtained", "Marks remaining"],
            datasets: [{ data: [Math.max(0, obtained), remaining], backgroundColor: ["#5969e8", "#e8ebf4"], borderWidth: 0, hoverOffset: 5 }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            cutout: "72%",
            plugins: { legend: { position: "bottom", labels: { usePointStyle: true, padding: 22 } } }
        }
    });
}

function downloadDetails() {
    if (!currentEvaluation) {
        alert("There is no evaluation report to download.");
        return;
    }
    const result = currentEvaluation;
    let report = "";
    report += "AI ANSWER SCRIPT EVALUATION\n";
    report += "====================================================\n\n";
    report += `Student Name: ${currentStudent.name}\n`;
    report += `USN: ${currentStudent.usn}\n`;
    report += `Reference Textbook: ${result.textbookFilename || textbookFilename || "Not specified"}\n\n`;
    report += "OVERALL RESULT\n";
    report += "----------------------------------------------------\n";
    report += `Marks: ${formatMarks(result.obtained)} / ${formatMarks(result.total)}\n`;
    report += `Percentage: ${result.percentage.toFixed(1)}%\n`;
    report += `Rating: ${getRating(result.percentage)}\n\n`;
    report += "QUESTION-WISE EVALUATION\n";
    report += "====================================================\n\n";

    result.questions.forEach(question => {
        report += `${question.question}\n`;
        if (question.questionText) report += `Question: ${question.questionText}\n`;
        report += `Maximum marks: ${formatMarks(question.max)}\n`;
        report += `Obtained marks: ${formatMarks(question.obtained)}\n`;
        if (question.studentAnswer) report += `Student answer: ${question.studentAnswer}\n`;
        if (question.modelAnswer) report += `Reference answer: ${question.modelAnswer}\n`;
        report += `Feedback: ${question.evaluation}\n`;
        if (question.textbookSupport) report += `Textbook support: ${question.textbookSupport}\n`;
        if (question.correctConcepts.length) report += `Correct concepts: ${question.correctConcepts.join(", ")}\n`;
        if (question.missingConcepts.length) report += `Missing concepts: ${question.missingConcepts.join(", ")}\n`;
        if (question.incorrectConcepts.length) report += `Incorrect concepts: ${question.incorrectConcepts.join(", ")}\n`;
        report += "\n----------------------------------------------------\n\n";
    });

    report += "\nGenerated by AI Answer Evaluator\n";
    const blob = new Blob([report], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `${safeFilename(currentStudent.usn || "student")}_evaluation_details.txt`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
}

function nextStudent() {
    document.getElementById("studentName").value = "";
    document.getElementById("studentUSN").value = "";
    document.getElementById("answerScript").value = "";
    document.getElementById("answerScriptName").textContent = "No answer script selected";
    currentStudent = { name: "", usn: "", answerScript: null };
    currentEvaluation = null;
    showPage("studentPage");
}

function exitEvaluation() {
    currentStudent = { name: "", usn: "", answerScript: null };
    currentEvaluation = null;
    showPage("referencePage");
}

function backToReference() {
    showPage("referencePage");
}

function logout() {
    answerKeyFile = null;
    textbookId = null;
    textbookFilename = "";
    currentStudent = { name: "", usn: "", answerScript: null };
    currentEvaluation = null;

    document.getElementById("loginForm").reset();
    document.getElementById("answerKey").value = "";
    document.getElementById("newTextbookFile").value = "";
    document.getElementById("answerScript").value = "";
    document.getElementById("answerKeyName").textContent = "No answer key selected";
    document.getElementById("newTextbookFileName").textContent = "No new textbook selected";
    document.getElementById("answerScriptName").textContent = "No answer script selected";
    document.getElementById("selectedAnswerKeySummary").textContent = "Upload an answer key PDF";
    document.getElementById("selectedTextbookSummary").textContent = "Select a textbook from your library";
    document.getElementById("studentName").value = "";
    document.getElementById("studentUSN").value = "";
    document.getElementById("textbookSelect").value = "";
    document.getElementById("textbookProcessStatus").textContent = "";
    document.getElementById("referenceStatus").classList.add("hidden");
    document.getElementById("selectedReferenceSummary").classList.remove("references-ready");
    showPage("loginPage");
}

function escapeHtml(value) {
    return String(value).replace(/[&<>"']/g, character => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
    })[character]);
}

function safeFilename(value) {
    return String(value).replace(/[^a-z0-9_-]/gi, "_");
}
