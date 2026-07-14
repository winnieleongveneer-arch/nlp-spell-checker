// -----------------------------
// DOM helpers
// -----------------------------
const $ = (sel) => document.querySelector(sel);

const editor = $("#editor");
const errorList = $("#errorList");
const selectedWordEl = $("#selectedWord");
const candidateListEl = $("#candidateList");
const btnReplace = $("#btnReplace");
const btnIgnore = $("#btnIgnore");

const charCounter = $("#charCounter");
const MAX_LEN = 500;

const saveModal = $("#saveModal");
const settingModal = $("#settingModal");

const fileInput = $("#fileInput");

const loadingOverlay = $("#loadingOverlay");
const selectedErrTypeEl = $("#selectedErrType");

// Save modal elements
const saveNameInput = $("#saveNameInput");
const btnSaveCurrent = $("#btnSaveCurrent");
const savedListEl = $("#savedList");

// -----------------------------
// State
// -----------------------------
let SETTINGS = {
  enable_auto_correction: false,
  show_candidate_ranking: true,
  //enable_finance_dictionary: false,
  show_confidence: false,
  enable_real_word_check: false
};

let LAST_CHECK = {
  original_text: "",
  errors: [],
  suggestions: {}
};

let UI_STATE = {
  selected_word: "",
  selected_candidate: "",
  selected_error_id: ""
};

// -----------------------------
// Loading overlay
// -----------------------------
function showLoading() {
  if (!loadingOverlay) return;
  loadingOverlay.classList.remove("hidden");
}

function hideLoading() {
  if (!loadingOverlay) return;
  loadingOverlay.classList.add("hidden");
}

async function apiFetch(url, options) {
  showLoading();
  try {
    return await fetch(url, options);
  } finally {
    hideLoading();
  }
}

// -----------------------------
// Modal helpers
// -----------------------------
function showModal(el) { if (el) el.classList.remove("hidden"); }
function hideModal(el) { if (el) el.classList.add("hidden"); }

function showSaveModal() { showModal(saveModal); }
function hideSaveModal() { hideModal(saveModal); }

function showSettingModal() {
  $("#optAutoCorrect").checked = SETTINGS.enable_auto_correction;
  $("#optShowRanking").checked = SETTINGS.show_candidate_ranking;
  //$("#optFinanceDict").checked = SETTINGS.enable_finance_dictionary;
  $("#optShowConfidence").checked = SETTINGS.show_confidence;
  $("#optRealWordCheck").checked = SETTINGS.enable_real_word_check;
  showModal(settingModal);
}

function hideSettingModal() { hideModal(settingModal); }

function syncSettingsFromUI() {
  SETTINGS.enable_auto_correction = $("#optAutoCorrect").checked;
  SETTINGS.show_candidate_ranking = $("#optShowRanking").checked;
  //SETTINGS.enable_finance_dictionary = $("#optFinanceDict").checked;
  SETTINGS.show_confidence = $("#optShowConfidence").checked;
  SETTINGS.enable_real_word_check = $("#optRealWordCheck").checked;
}

// -----------------------------
// Utility
// -----------------------------
function escapeHtml(str) {
  return String(str || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function getPlainText() {
  return editor.innerText || "";
}

function setPlainText(t) {
  t = String(t || "");
  if (t.length > MAX_LEN) t = t.slice(0, MAX_LEN);
  editor.innerText = t;
  updateCharCounter(t.length);
}

function updateCharCounter(len) {
  if (!charCounter) return;
  charCounter.textContent = `${len}/${MAX_LEN}`;
  if (len >= MAX_LEN) charCounter.classList.add("limit-hit");
  else charCounter.classList.remove("limit-hit");
}

// -----------------------------
// 500 character hard-limit (block insertion, allow deletion)
// -----------------------------
function isSelectionInsideEditor() {
  const sel = window.getSelection();
  if (!sel || sel.rangeCount === 0) return false;
  const range = sel.getRangeAt(0);
  return editor.contains(range.startContainer) && editor.contains(range.endContainer);
}

editor.addEventListener("beforeinput", (e) => {
  const inputType = e.inputType || "";

  // Always allow deletions
  if (inputType.startsWith("delete")) return;

  // Only enforce when selection is inside editor
  if (!isSelectionInsideEditor()) return;

  const currentText = getPlainText();
  const currentLen = currentText.length;

  // Selected length (replacement scenario)
  const sel = window.getSelection();
  const selectedLen = sel ? (sel.toString() || "").length : 0;

  // Allowed remaining after considering replacement
  const remaining = MAX_LEN - (currentLen - selectedLen);

  // If no remaining space, block all insert-type inputs
  if (remaining <= 0) {
    e.preventDefault();
    return;
  }

  // Handle paste explicitly (insertFromPaste)
  if (inputType === "insertFromPaste") {
    const pasteText = (e.clipboardData && e.clipboardData.getData("text/plain")) || "";
    if (!pasteText) return;

    e.preventDefault();
    const truncated = pasteText.slice(0, remaining);

    // Insert the truncated text at caret
    document.execCommand("insertText", false, truncated);
    return;
  }

  // For normal typing/composition
  const data = typeof e.data === "string" ? e.data : "";
  if (data && data.length > remaining) {
    e.preventDefault();
    document.execCommand("insertText", false, data.slice(0, remaining));
  }
});

editor.addEventListener("input", () => {
  const len = getPlainText().length;
  updateCharCounter(Math.min(len, MAX_LEN));
});

// -----------------------------
// Error list rendering
// -----------------------------
function renderErrorList(errors) {
  errorList.innerHTML = "";
  if (!errors || errors.length === 0) {
    const li = document.createElement("li");
    li.textContent = "No errors";
    li.classList.add("empty");
    errorList.appendChild(li);
    return;
  }

  errors.forEach((e, idx) => {
    const li = document.createElement("li");
    li.textContent = `${idx + 1}. ${e.word}`;
    li.dataset.index = String(idx);
    li.classList.add(e.type === "real-word" ? "rw" : "nw");
    li.addEventListener("click", () => {
      highlightErrorInEditor(idx, errors);
      showSuggestionsForError(e);
    });
    errorList.appendChild(li);
  });
}

// -----------------------------
// Suggestions UI
// -----------------------------
function clearSuggestionsUI() {
  UI_STATE.selected_word = "";
  UI_STATE.selected_candidate = "";
  UI_STATE.selected_error_id = "";
  selectedWordEl.textContent = "-";

  if (selectedErrTypeEl) {
    selectedErrTypeEl.textContent = "-";
    selectedErrTypeEl.classList.remove("nonword", "realword");
  }

  candidateListEl.innerHTML = "";
  btnReplace.disabled = true;
  btnIgnore.disabled = true;
}

function showSuggestionsForError(er) {
  if (!er) return;
  const word = er.word || "";
  const errorId = er.id || "";

  UI_STATE.selected_word = word;
  UI_STATE.selected_candidate = "";
  UI_STATE.selected_error_id = errorId;

  selectedWordEl.textContent = word;

  // show error type
  if (selectedErrTypeEl) {
    const isRealWord = (er.type === "real-word");
    selectedErrTypeEl.textContent = isRealWord ? "Real-word error" : "Non-word error";
    selectedErrTypeEl.classList.remove("nonword", "realword");
    selectedErrTypeEl.classList.add(isRealWord ? "realword" : "nonword");
  }

  candidateListEl.innerHTML = "";

  const sugMap = (LAST_CHECK && LAST_CHECK.suggestions) || {};
  const directSuggs = Array.isArray(er.candidates) ? er.candidates : (Array.isArray(er.suggestions) ? er.suggestions : null);
  const suggs = directSuggs || (errorId && sugMap[errorId]) || sugMap[word] || [];

  if (!suggs || suggs.length === 0) {
    const li = document.createElement("li");
    li.textContent = "No candidates";
    li.classList.add("empty");
    candidateListEl.appendChild(li);
    btnReplace.disabled = true;
    btnIgnore.disabled = false;
    return;
  }

  suggs.forEach((s) => {
    const li = document.createElement("li");
    li.classList.add("candidate-item");
    const cand = s.candidate || s.token || "";
    li.dataset.candidate = cand;

    if (!cand) {
      li.textContent = "Invalid candidate";
    } else {
      const parts = [];

      // ED
      if (SETTINGS.show_candidate_ranking && Number.isFinite(s.edit_distance)) {
        parts.push(`ED=${s.edit_distance}`);
      }

      // confidence %
      if (SETTINGS.show_confidence && typeof s.score === "number") {
        parts.push(`p=${(s.score * 100).toFixed(1)}%`);
      } else if (SETTINGS.show_candidate_ranking && typeof s.score === "number" && !Number.isFinite(s.edit_distance)) {
        parts.push(`score=${s.score.toFixed(3)}`);
      }

      const extra = parts.length ? ` (${parts.join(", ")})` : "";
      li.textContent = `${cand}${extra}`;
    }

    candidateListEl.appendChild(li);
  });

  btnIgnore.disabled = false;
  btnReplace.disabled = true;
}

// Candidate click (event delegation)
candidateListEl.addEventListener("click", (e) => {
  const li = e.target.closest("li");
  if (!li || !candidateListEl.contains(li)) return;
  if (li.classList.contains("empty")) return;

  candidateListEl.querySelectorAll("li").forEach((x) => x.classList.remove("active"));
  li.classList.add("active");

  UI_STATE.selected_candidate = li.dataset.candidate || "";
  btnReplace.disabled = !UI_STATE.selected_candidate;
  btnIgnore.disabled = false;
});

// -----------------------------
// Highlight rendering
// -----------------------------
function renderWithHighlights(text, errors) {
  if (!errors || errors.length === 0) {
    editor.textContent = text;
    updateCharCounter(text.length);
    return;
  }

  const sorted = [...errors].sort((a, b) => a.start - b.start);

  let html = "";
  let cursor = 0;

  for (const er of sorted) {
    const before = text.slice(cursor, er.start);
    const word = text.slice(er.start, er.end);

    html += escapeHtml(before);
    const kindClass = (er.type === "real-word") ? "realword" : "nonword";
    html += `<span class="err ${kindClass}" data-start="${er.start}" data-end="${er.end}">${escapeHtml(word)}</span>`;
    cursor = er.end;
  }

  html += escapeHtml(text.slice(cursor));
  editor.innerHTML = html;
  updateCharCounter(text.length);
}

function highlightErrorInEditor(errorIndex, errors) {
  const er = errors[errorIndex];
  if (!er) return;

  const spans = editor.querySelectorAll("span.err");
  spans.forEach((s) => s.classList.remove("active"));

  for (const s of spans) {
    const start = Number(s.dataset.start);
    const end = Number(s.dataset.end);
    if (start === er.start && end === er.end) {
      s.classList.add("active");
      break;
    }
  }
}

// Click highlighted word
editor.addEventListener("click", (e) => {
  const target = e.target;
  if (target && target.tagName === "SPAN" && target.classList.contains("err")) {
    const start = Number(target.dataset.start);
    const end = Number(target.dataset.end);
    const errors = LAST_CHECK.errors || [];
    const idx = errors.findIndex((x) => x.start === start && x.end === end);
    if (idx !== -1) {
      highlightErrorInEditor(idx, errors);
      showSuggestionsForError(errors[idx]);
    }
  }
});

// -----------------------------
// Save: load / render list
// -----------------------------
async function loadSavedTexts() {
  if (!savedListEl) return;

  const resp = await fetch("/api/saved");
  if (!resp.ok) return;

  const data = await resp.json();
  const items = data.items || [];

  savedListEl.innerHTML = "";

  if (!items.length) {
    const empty = document.createElement("div");
    empty.className = "saved-item";
    empty.style.opacity = "0.7";
    empty.textContent = "No saved text yet.";
    savedListEl.appendChild(empty);
    return;
  }

  items.forEach(item => {
    const row = document.createElement("div");
    row.className = "saved-item";
    row.dataset.id = item.id;

    const left = document.createElement("div");
    left.className = "saved-item-text";
    left.textContent = item.preview || item.name || item.id;

    const del = document.createElement("button");
    del.className = "saved-item-del";
    del.textContent = "Delete";
    del.dataset.id = item.id;

    row.appendChild(left);
    row.appendChild(del);
    savedListEl.appendChild(row);
  });
}

// Save list click: load / delete
if (savedListEl) {
  savedListEl.addEventListener("click", async (e) => {
    const delBtn = e.target.closest(".saved-item-del");
    if (delBtn) {
      const id = delBtn.dataset.id;
      if (!confirm("Delete this saved text?")) return;

      const resp = await fetch(`/api/saved/${id}`, { method: "DELETE" });
      if (resp.ok) await loadSavedTexts();
      return;
    }

    const row = e.target.closest(".saved-item");
    if (!row || !row.dataset.id) return;

    const id = row.dataset.id;
    const resp = await fetch(`/api/saved/${id}`);
    if (!resp.ok) return;

    const data = await resp.json();
    setPlainText((data.text || "").slice(0, MAX_LEN));
    hideSaveModal();
  });
}

// -----------------------------
// Buttons
// -----------------------------
$("#btnUpload")?.addEventListener("click", () => fileInput.click());

fileInput?.addEventListener("change", async () => {
  const f = fileInput.files[0];
  if (!f) return;

  const form = new FormData();
  form.append("file", f);

  const res = await apiFetch("/api/upload", { method: "POST", body: form });
  const data = await res.json();

  if (data && data.text) {
    setPlainText((data.text || "").slice(0, MAX_LEN));
    renderErrorList([]);
    clearSuggestionsUI();
  }
});

// Save text: open modal ONLY (do not auto-save)
$("#btnSaveText")?.addEventListener("click", async () => {
  showSaveModal();
  await loadSavedTexts();
});

// Save current: actually save
btnSaveCurrent?.addEventListener("click", async () => {
  const text = getPlainText().slice(0, MAX_LEN);
  const name = (saveNameInput?.value || "").trim();
  const payload = name ? { text, name } : { text };

  const resp = await fetch("/api/save", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });

  if (!resp.ok) {
    alert("Save failed.");
    return;
  }

  if (saveNameInput) saveNameInput.value = "";
  await loadSavedTexts();
});

$("#btnSetting")?.addEventListener("click", showSettingModal);
$("#btnSaveBack")?.addEventListener("click", hideSaveModal);

$("#btnSettingSave")?.addEventListener("click", () => {
  syncSettingsFromUI();
  hideSettingModal();
});

$("#btnSettingCancel")?.addEventListener("click", hideSettingModal);

$("#btnClear")?.addEventListener("click", () => {
  setPlainText("");
  renderErrorList([]);
  clearSuggestionsUI();
});

$("#btnExport")?.addEventListener("click", async () => {
  const text = getPlainText().slice(0, MAX_LEN);

  const res = await apiFetch("/api/export", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text })
  });

  if (!res.ok) {
    alert("Export failed");
    return;
  }

  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "spellchecker_output.txt";
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
});

// -----------------------------
// Run check (with overlay)
// -----------------------------
async function runCheck() {
  const text = getPlainText().slice(0, MAX_LEN);

  if (!text.trim()) {
    renderErrorList([]);
    clearSuggestionsUI();
    return;
  }

  const payload = { text, settings: { ...SETTINGS } };

  const res = await apiFetch("/api/check", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });

  const data = await res.json();

  LAST_CHECK = data || { errors: [], suggestions: {} };
  LAST_CHECK.original_text = (data && (data.original_text ?? data.text)) ?? text;

  renderErrorList(LAST_CHECK.errors || []);
  clearSuggestionsUI();
  renderWithHighlights(text, LAST_CHECK.errors || []);

  if ((LAST_CHECK.errors || []).length > 0) {
    highlightErrorInEditor(0, LAST_CHECK.errors);
    showSuggestionsForError(LAST_CHECK.errors[0]);
  }

  // Optional auto-correct
  if (SETTINGS.enable_auto_correction && (LAST_CHECK.errors || []).length > 0) {
    const res2 = await apiFetch("/api/autocorrect", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text,
        errors: LAST_CHECK.errors,
        suggestions: LAST_CHECK.suggestions
      })
    });

    const data2 = await res2.json();
    if (data2 && data2.text) {
      setPlainText((data2.text || "").slice(0, MAX_LEN));

      // Re-run check once after autocorrect, but disable autocorrect to avoid loops
      const payload2 = {
        text: getPlainText().slice(0, MAX_LEN),
        settings: { ...SETTINGS, enable_auto_correction: false }
      };

      const res3 = await apiFetch("/api/check", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload2)
      });

      const data3 = await res3.json();

      LAST_CHECK = data3 || { errors: [], suggestions: {} };
      const text3 = getPlainText().slice(0, MAX_LEN);
      LAST_CHECK.original_text = (data3 && (data3.original_text ?? data3.text)) ?? text3;

      renderErrorList(LAST_CHECK.errors || []);
      clearSuggestionsUI();
      renderWithHighlights(text3, LAST_CHECK.errors || []);

      if ((LAST_CHECK.errors || []).length > 0) {
        highlightErrorInEditor(0, LAST_CHECK.errors);
        showSuggestionsForError(LAST_CHECK.errors[0]);
      }
    }
  }
}

$("#btnRunCheck")?.addEventListener("click", runCheck);

// -----------------------------
// Replace / Ignore
// -----------------------------
btnReplace?.addEventListener("click", async () => {
  const word = UI_STATE.selected_word || selectedWordEl.textContent;
  const candidate = UI_STATE.selected_candidate;

  if (!word || !candidate) return;

  const errors = LAST_CHECK.errors || [];
  const errId = UI_STATE.selected_error_id;
  const err = (errId && errors.find((e) => e.id === errId)) || errors.find((e) => e.word === word);
  if (!err) return;

  const baseText = (LAST_CHECK.original_text || getPlainText()).slice(0, MAX_LEN);

  const before = baseText.slice(0, err.start);
  const after = baseText.slice(err.end);
  const newText = (before + candidate + after).slice(0, MAX_LEN);

  setPlainText(newText);
  renderErrorList([]);
  clearSuggestionsUI();

  // Re-run check
  const payload = { text: getPlainText().slice(0, MAX_LEN), settings: { ...SETTINGS } };
  const res = await apiFetch("/api/check", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });

  const data = await res.json();

  LAST_CHECK = data || { errors: [], suggestions: {} };
  const textNow = getPlainText().slice(0, MAX_LEN);
  LAST_CHECK.original_text = (data && (data.original_text ?? data.text)) ?? textNow;

  renderErrorList(LAST_CHECK.errors || []);
  clearSuggestionsUI();
  renderWithHighlights(textNow, LAST_CHECK.errors || []);
});

btnIgnore?.addEventListener("click", () => {
  const id = UI_STATE.selected_error_id;
  if (!id || !LAST_CHECK || !Array.isArray(LAST_CHECK.errors)) {
    clearSuggestionsUI();
    return;
  }

  const idx = LAST_CHECK.errors.findIndex(e => e && e.id === id);
  if (idx !== -1) LAST_CHECK.errors.splice(idx, 1);

  if (LAST_CHECK.suggestions && typeof LAST_CHECK.suggestions === "object") {
    delete LAST_CHECK.suggestions[id];
  }

  clearSuggestionsUI();
  const textNow = (LAST_CHECK.original_text ?? getPlainText()).slice(0, MAX_LEN);
  renderErrorList(LAST_CHECK.errors);
  renderWithHighlights(textNow, LAST_CHECK.errors);
});

// -----------------------------
// Init
// -----------------------------
document.addEventListener("DOMContentLoaded", async () => {
  renderErrorList([]);
  clearSuggestionsUI();
  updateCharCounter(getPlainText().length);
  await loadSavedTexts();
});
