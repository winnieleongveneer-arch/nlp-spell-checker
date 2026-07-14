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

const mainGridEl = $("#mainGrid");
const vocabPanelEl = $("#vocabPanel");
const vocabSearchEl = $("#vocabSearch");
const vocabListEl = $("#vocabList");
const vocabCountEl = $("#vocabCount");

const saveNameInput = $("#saveNameInput");
const btnSaveCurrent = $("#btnSaveCurrent");
const savedListEl = $("#savedList");

let SETTINGS = {
  enable_auto_correction: false,
  show_candidate_ranking: true,
  show_confidence: false,
  enable_real_word_check: false,
  show_vocab_panel: false
};

let LAST_CHECK = { original_text: "", errors: [], suggestions: {} };

let UI_STATE = {
  selected_word: "",
  selected_candidate: "",
  selected_error_id: ""
};

function showLoading() { loadingOverlay?.classList.remove("hidden"); }
function hideLoading() { loadingOverlay?.classList.add("hidden"); }

async function apiFetch(url, options) {
  showLoading();
  try { return await fetch(url, options); }
  finally { hideLoading(); }
}

function showModal(el){ el?.classList.remove("hidden"); }
function hideModal(el){ el?.classList.add("hidden"); }

function showSaveModal(){ showModal(saveModal); }
function hideSaveModal(){ hideModal(saveModal); }

function showSettingModal(){
  $("#optAutoCorrect").checked = SETTINGS.enable_auto_correction;
  $("#optShowRanking").checked = SETTINGS.show_candidate_ranking;
  $("#optShowConfidence").checked = SETTINGS.show_confidence;
  $("#optRealWordCheck").checked = SETTINGS.enable_real_word_check;
  $("#optShowVocabPanel").checked = SETTINGS.show_vocab_panel;
  showModal(settingModal);
}
function hideSettingModal(){ hideModal(settingModal); }

function syncSettingsFromUI(){
  SETTINGS.enable_auto_correction = $("#optAutoCorrect").checked;
  SETTINGS.show_candidate_ranking = $("#optShowRanking").checked;
  SETTINGS.show_confidence = $("#optShowConfidence").checked;
  SETTINGS.enable_real_word_check = $("#optRealWordCheck").checked;
  SETTINGS.show_vocab_panel = $("#optShowVocabPanel").checked;
}

function escapeHtml(str){
  return String(str||"")
    .replace(/&/g,"&amp;").replace(/</g,"&lt;")
    .replace(/>/g,"&gt;");
}

function getPlainText(){ return editor.innerText || ""; }

function setPlainText(t){
  t = String(t||"");
  if (t.length > MAX_LEN) t = t.slice(0, MAX_LEN);
  editor.innerText = t;
  updateCharCounter(t.length);
}

function updateCharCounter(len){
  if (!charCounter) return;
  charCounter.textContent = `${len}/${MAX_LEN}`;
  if (len >= MAX_LEN) charCounter.classList.add("limit-hit");
  else charCounter.classList.remove("limit-hit");
}

/* 500 limit */
function isSelectionInsideEditor(){
  const sel = window.getSelection();
  if (!sel || sel.rangeCount === 0) return false;
  const range = sel.getRangeAt(0);
  return editor.contains(range.startContainer) && editor.contains(range.endContainer);
}

editor.addEventListener("beforeinput", (e) => {
  const inputType = e.inputType || "";
  if (inputType.startsWith("delete")) return;
  if (!isSelectionInsideEditor()) return;

  const currentText = getPlainText();
  const currentLen = currentText.length;
  const sel = window.getSelection();
  const selectedLen = sel ? (sel.toString() || "").length : 0;
  const remaining = MAX_LEN - (currentLen - selectedLen);

  if (remaining <= 0) { e.preventDefault(); return; }

  if (inputType === "insertFromPaste") {
    const pasteText = (e.clipboardData && e.clipboardData.getData("text/plain")) || "";
    if (!pasteText) return;
    e.preventDefault();
    document.execCommand("insertText", false, pasteText.slice(0, remaining));
    return;
  }

  const data = typeof e.data === "string" ? e.data : "";
  if (data && data.length > remaining) {
    e.preventDefault();
    document.execCommand("insertText", false, data.slice(0, remaining));
  }
});

editor.addEventListener("input", () => updateCharCounter(getPlainText().length));

function renderErrorList(errors){
  errorList.innerHTML = "";
  if (!errors || errors.length === 0){
    const li = document.createElement("li");
    li.textContent = "No errors";
    li.classList.add("empty");
    errorList.appendChild(li);
    return;
  }
  errors.forEach((e, idx) => {
    const li = document.createElement("li");
    li.textContent = `${idx+1}. ${e.word}`;
    li.dataset.index = String(idx);
    li.classList.add(e.type === "real-word" ? "rw" : "nw");
    li.addEventListener("click", () => {
      highlightErrorInEditor(idx, errors);
      showSuggestionsForError(e);
    });
    errorList.appendChild(li);
  });
}

function clearSuggestionsUI(){
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

function showSuggestionsForError(er){
  if (!er) return;
  const word = er.word || "";
  const errorId = er.id || "";

  UI_STATE.selected_word = word;
  UI_STATE.selected_candidate = "";
  UI_STATE.selected_error_id = errorId;

  selectedWordEl.textContent = word;

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

    const parts = [];
    if (SETTINGS.show_candidate_ranking && Number.isFinite(s.edit_distance)) parts.push(`ED=${s.edit_distance}`);
    if (SETTINGS.show_confidence && typeof s.score === "number") parts.push(`p=${(s.score*100).toFixed(1)}%`);

    const extra = parts.length ? ` (${parts.join(", ")})` : "";
    li.textContent = `${cand}${extra}`;
    candidateListEl.appendChild(li);
  });

  btnIgnore.disabled = false;
  btnReplace.disabled = true;
}

candidateListEl.addEventListener("click", (e) => {
  const li = e.target.closest("li");
  if (!li || !candidateListEl.contains(li) || li.classList.contains("empty")) return;
  candidateListEl.querySelectorAll("li").forEach((x) => x.classList.remove("active"));
  li.classList.add("active");
  UI_STATE.selected_candidate = li.dataset.candidate || "";
  btnReplace.disabled = !UI_STATE.selected_candidate;
  btnIgnore.disabled = false;
});

function renderWithHighlights(text, errors){
  if (!errors || errors.length === 0) {
    editor.textContent = text;
    updateCharCounter(text.length);
    return;
  }
  const sorted = [...errors].sort((a,b)=>a.start-b.start);
  let html = "";
  let cursor = 0;
  for (const er of sorted){
    html += escapeHtml(text.slice(cursor, er.start));
    const kindClass = (er.type === "real-word") ? "realword" : "nonword";
    html += `<span class="err ${kindClass}" data-start="${er.start}" data-end="${er.end}">${escapeHtml(text.slice(er.start, er.end))}</span>`;
    cursor = er.end;
  }
  html += escapeHtml(text.slice(cursor));
  editor.innerHTML = html;
  updateCharCounter(text.length);
}

function highlightErrorInEditor(errorIndex, errors){
  const er = errors[errorIndex];
  if (!er) return;
  editor.querySelectorAll("span.err").forEach(s => s.classList.remove("active"));
  for (const s of editor.querySelectorAll("span.err")) {
    if (Number(s.dataset.start) === er.start && Number(s.dataset.end) === er.end) {
      s.classList.add("active"); break;
    }
  }
}

editor.addEventListener("click", (e) => {
  const t = e.target;
  if (t && t.tagName === "SPAN" && t.classList.contains("err")) {
    const start = Number(t.dataset.start);
    const end = Number(t.dataset.end);
    const errors = LAST_CHECK.errors || [];
    const idx = errors.findIndex((x)=>x.start===start && x.end===end);
    if (idx !== -1){
      highlightErrorInEditor(idx, errors);
      showSuggestionsForError(errors[idx]);
    }
  }
});

/* ===========================
   Vocabulary panel (性能优化版)
   =========================== */
function applyVocabPanelVisibility(){
  if (!mainGridEl || !vocabPanelEl) return;

  if (SETTINGS.show_vocab_panel){
    mainGridEl.classList.add("with-vocab");
    vocabPanelEl.classList.remove("hidden");
    loadVocabWords(); // 只在显示时加载
  } else {
    mainGridEl.classList.remove("with-vocab");
    vocabPanelEl.classList.add("hidden");
  }
}

let vocabSearchTimer = null;
let vocabCache = { key: "", words: [], rendered: 0 };

const VOCAB_BATCH = 400;
const VOCAB_NEAR_BOTTOM = 80;

function setVocabLoading(){
  if (!vocabListEl) return;
  vocabListEl.innerHTML = `<div class="vocab-empty">Loading...</div>`;
}

function updateVocabCount(){
  if (!vocabCountEl) return;
  const total = vocabCache.words.length || 0;
  vocabCountEl.textContent = `${total} words`;
}

function appendVocabBatch(){
  if (!vocabListEl) return;
  const words = vocabCache.words || [];
  if (vocabCache.rendered >= words.length) return;

  const next = words.slice(vocabCache.rendered, vocabCache.rendered + VOCAB_BATCH);
  vocabCache.rendered += next.length;

  const frag = document.createDocumentFragment();
  for (const w of next) {
    const item = document.createElement("div");
    item.className = "vocab-item";
    item.textContent = w;
    frag.appendChild(item);
  }
  vocabListEl.appendChild(frag);
}

function resetVocabRender(words){
  vocabCache.words = Array.isArray(words) ? words : [];
  vocabCache.rendered = 0;

  vocabListEl.innerHTML = "";
  updateVocabCount();

  if (!vocabCache.words.length) {
    vocabListEl.innerHTML = `<div class="vocab-empty">No results</div>`;
    return;
  }

  appendVocabBatch();
}

async function loadVocabWords(){
  if (!SETTINGS.show_vocab_panel) return;
  if (!vocabPanelEl || vocabPanelEl.classList.contains("hidden")) return;

  const q = (vocabSearchEl?.value || "").trim().toLowerCase();
  const key = q;

  if (vocabCache.key === key && vocabCache.words.length) return;
  vocabCache.key = key;

  setVocabLoading();

  // ✅ 固定使用 finance extra（你不需要前端 checkbox 了）
  const url = `/api/words?search=${encodeURIComponent(q)}&use_fin=true`;

  try {
    const resp = await fetch(url);
    if (!resp.ok) {
      vocabListEl.innerHTML = `<div class="vocab-empty">Failed to load</div>`;
      return;
    }
    const data = await resp.json();
    const words = data.words || [];
    resetVocabRender(words);
  } catch {
    vocabListEl.innerHTML = `<div class="vocab-empty">Failed to load</div>`;
  }
}

vocabListEl?.addEventListener("scroll", () => {
  if (!vocabCache.words.length) return;
  const nearBottom = (vocabListEl.scrollTop + vocabListEl.clientHeight) >= (vocabListEl.scrollHeight - VOCAB_NEAR_BOTTOM);
  if (nearBottom) appendVocabBatch();
});

vocabSearchEl?.addEventListener("input", () => {
  if (vocabSearchTimer) clearTimeout(vocabSearchTimer);
  vocabSearchTimer = setTimeout(() => {
    vocabCache.key = ""; // force reload
    loadVocabWords();
  }, 180);
});

/* ===========================
   Save / Load list (unchanged)
   =========================== */
async function loadSavedTexts(){
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

savedListEl?.addEventListener("click", async (e) => {
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

/* Buttons */
$("#btnUpload")?.addEventListener("click", () => fileInput.click());

fileInput?.addEventListener("change", async () => {
  const f = fileInput.files[0];
  if (!f) return;
  const form = new FormData();
  form.append("file", f);
  const res = await apiFetch("/api/upload", { method:"POST", body: form });
  const data = await res.json();
  if (data?.text) {
    setPlainText(data.text.slice(0, MAX_LEN));
    renderErrorList([]);
    clearSuggestionsUI();
  }
});

$("#btnSaveText")?.addEventListener("click", async () => {
  showSaveModal();
  await loadSavedTexts();
});

btnSaveCurrent?.addEventListener("click", async () => {
  const text = getPlainText().slice(0, MAX_LEN);
  const name = (saveNameInput?.value || "").trim();
  const payload = name ? { text, name } : { text };

  const resp = await fetch("/api/save", {
    method:"POST",
    headers:{ "Content-Type":"application/json" },
    body: JSON.stringify(payload)
  });
  if (!resp.ok) { alert("Save failed."); return; }
  if (saveNameInput) saveNameInput.value = "";
  await loadSavedTexts();
});

$("#btnSetting")?.addEventListener("click", showSettingModal);
$("#btnSaveBack")?.addEventListener("click", hideSaveModal);

$("#btnSettingSave")?.addEventListener("click", () => {
  syncSettingsFromUI();
  hideSettingModal();
  applyVocabPanelVisibility();
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
    method:"POST",
    headers:{ "Content-Type":"application/json" },
    body: JSON.stringify({ text })
  });
  if (!res.ok) { alert("Export failed"); return; }
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

async function runCheck(){
  const text = getPlainText().slice(0, MAX_LEN);
  if (!text.trim()){
    renderErrorList([]);
    clearSuggestionsUI();
    return;
  }
  const payload = { text, settings: { ...SETTINGS } };
  const res = await apiFetch("/api/check", {
    method:"POST",
    headers:{ "Content-Type":"application/json" },
    body: JSON.stringify(payload)
  });
  const data = await res.json();
  LAST_CHECK = data || { errors:[], suggestions:{} };
  LAST_CHECK.original_text = (data && (data.original_text ?? data.text)) ?? text;

  renderErrorList(LAST_CHECK.errors || []);
  clearSuggestionsUI();
  renderWithHighlights(text, LAST_CHECK.errors || []);

  if ((LAST_CHECK.errors || []).length > 0) {
    highlightErrorInEditor(0, LAST_CHECK.errors);
    showSuggestionsForError(LAST_CHECK.errors[0]);
  }
}

$("#btnRunCheck")?.addEventListener("click", runCheck);

btnReplace?.addEventListener("click", async () => {
  const word = UI_STATE.selected_word || selectedWordEl.textContent;
  const candidate = UI_STATE.selected_candidate;
  if (!word || !candidate) return;

  const errors = LAST_CHECK.errors || [];
  const errId = UI_STATE.selected_error_id;
  const err = (errId && errors.find(e=>e.id===errId)) || errors.find(e=>e.word===word);
  if (!err) return;

  const baseText = (LAST_CHECK.original_text || getPlainText()).slice(0, MAX_LEN);
  const newText = (baseText.slice(0, err.start) + candidate + baseText.slice(err.end)).slice(0, MAX_LEN);

  setPlainText(newText);
  renderErrorList([]);
  clearSuggestionsUI();
  await runCheck();
});

btnIgnore?.addEventListener("click", () => {
  const id = UI_STATE.selected_error_id;
  if (!id || !Array.isArray(LAST_CHECK.errors)) { clearSuggestionsUI(); return; }

  const idx = LAST_CHECK.errors.findIndex(e=>e?.id===id);
  if (idx !== -1) LAST_CHECK.errors.splice(idx, 1);
  if (LAST_CHECK.suggestions && typeof LAST_CHECK.suggestions === "object") delete LAST_CHECK.suggestions[id];

  clearSuggestionsUI();
  const textNow = (LAST_CHECK.original_text ?? getPlainText()).slice(0, MAX_LEN);
  renderErrorList(LAST_CHECK.errors);
  renderWithHighlights(textNow, LAST_CHECK.errors);
});

document.addEventListener("DOMContentLoaded", async () => {
  renderErrorList([]);
  clearSuggestionsUI();
  updateCharCounter(getPlainText().length);
  await loadSavedTexts();

  // ✅ 关键：按 SETTINGS.show_vocab_panel 决定是否显示
  applyVocabPanelVisibility();
});
