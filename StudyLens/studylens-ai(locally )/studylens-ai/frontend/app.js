(function () {
  const state = {
    imageBase64: null,
    imageMimeType: null,
    history: [], // Ollama-format messages: [{role:'user'|'assistant', content:'...', images?:[...]}]
    busy: false,
  };

  const els = {
    dropzone: document.getElementById("dropzone"),
    cameraBtn: document.getElementById("camera-btn"),
    galleryBtn: document.getElementById("gallery-btn"),
    cameraInput: document.getElementById("camera-input"),
    galleryInput: document.getElementById("gallery-input"),
    previewWrap: document.getElementById("preview-wrap"),
    previewImg: document.getElementById("preview-img"),
    previewTitle: document.getElementById("preview-title"),
    previewSub: document.getElementById("preview-sub"),
    errorContainer: document.getElementById("error-container"),
    analysisSection: document.getElementById("analysis-section"),
    typePill: document.getElementById("type-pill"),
    resetBtn: document.getElementById("reset-btn"),
    outSummary: document.getElementById("out-summary"),
    outConcepts: document.getElementById("out-concepts"),
    outTerms: document.getElementById("out-terms"),
    outExplanation: document.getElementById("out-explanation"),
    outQuestions: document.getElementById("out-questions"),
    quickActions: document.getElementById("quick-actions"),
    chatSection: document.getElementById("chat-section"),
    chatLog: document.getElementById("chat-log"),
    chatForm: document.getElementById("chat-form"),
    chatInput: document.getElementById("chat-input"),
    chatSend: document.getElementById("chat-send"),
  };

  const TYPE_LABELS = {
    book: "📖 صفحة كتاب",
    lecture: "🎓 محاضرة",
    chart: "📊 رسم بياني",
    equation: "🧮 معادلة",
    table: "📋 جدول",
    diagram: "🗺️ مخطط",
    other: "📄 محتوى تعليمي",
  };

  function escapeHtml(s) {
    return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }

  function inlineMd(s) {
    s = escapeHtml(s);
    s = s.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
    s = s.replace(/\*(.+?)\*/g, "<em>$1</em>");
    return s;
  }

  function mdToHtml(text) {
    const lines = String(text || "").split("\n");
    let html = "";
    let listMode = false;
    for (const raw of lines) {
      const line = raw.trim();
      if (/^[-*]\s+/.test(line)) {
        if (listMode !== "ul") {
          if (listMode) html += listMode === "ol" ? "</ol>" : "</ul>";
          html += "<ul>";
          listMode = "ul";
        }
        html += `<li>${inlineMd(line.replace(/^[-*]\s+/, ""))}</li>`;
      } else if (/^\d+\.\s+/.test(line)) {
        if (listMode !== "ol") {
          if (listMode) html += listMode === "ol" ? "</ol>" : "</ul>";
          html += "<ol>";
          listMode = "ol";
        }
        html += `<li>${inlineMd(line.replace(/^\d+\.\s+/, ""))}</li>`;
      } else {
        if (listMode) {
          html += listMode === "ol" ? "</ol>" : "</ul>";
          listMode = false;
        }
        if (line === "") continue;
        html += `<p>${inlineMd(line)}</p>`;
      }
    }
    if (listMode) html += listMode === "ol" ? "</ol>" : "</ul>";
    return html || "<p></p>";
  }

  function showError(msg) {
    els.errorContainer.innerHTML = `<div class="error-box">${escapeHtml(msg)}</div>`;
  }
  function clearError() {
    els.errorContainer.innerHTML = "";
  }

  function resizeImageFile(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onerror = () => reject(new Error("تعذّرت قراءة الملف."));
      reader.onload = (e) => {
        const img = new Image();
        img.onerror = () => reject(new Error("تعذّرت قراءة الصورة."));
        img.onload = () => {
          let { width, height } = img;
          const maxDim = 1024; // أصغر قليلاً من السحابي لتسريع الاستدلال المحلي
          if (width > maxDim || height > maxDim) {
            if (width > height) {
              height = Math.round((height * maxDim) / width);
              width = maxDim;
            } else {
              width = Math.round((width * maxDim) / height);
              height = maxDim;
            }
          }
          const canvas = document.createElement("canvas");
          canvas.width = width;
          canvas.height = height;
          canvas.getContext("2d").drawImage(img, 0, 0, width, height);
          const dataUrl = canvas.toDataURL("image/jpeg", 0.85);
          resolve({ dataUrl, base64: dataUrl.split(",")[1], mimeType: "image/jpeg" });
        };
        img.src = e.target.result;
      };
      reader.readAsDataURL(file);
    });
  }

  async function postJson(url, body) {
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    let payload;
    try {
      payload = await res.json();
    } catch (e) {
      throw new Error("تعذّر فهم رد الخادم المحلي.");
    }
    if (!payload.ok) {
      throw new Error(payload.error || "حدث خطأ غير متوقع.");
    }
    return payload;
  }

  function renderAnalysis(data) {
    els.typePill.textContent = TYPE_LABELS[data.content_type] || TYPE_LABELS.other;
    els.outSummary.textContent = data.summary || "—";

    els.outConcepts.innerHTML = "";
    (data.key_concepts || []).forEach((c) => {
      const span = document.createElement("span");
      span.className = "chip";
      span.textContent = c;
      els.outConcepts.appendChild(span);
    });

    els.outTerms.innerHTML = "";
    (data.important_terms || []).forEach((t) => {
      const div = document.createElement("div");
      div.className = "term-item";
      div.innerHTML = `<div class="term">${escapeHtml(t.term || "")}</div><div class="def">${escapeHtml(
        t.definition || ""
      )}</div>`;
      els.outTerms.appendChild(div);
    });

    els.outExplanation.textContent = data.explanation || "—";

    els.outQuestions.innerHTML = "";
    (data.questions || []).forEach((q) => {
      const li = document.createElement("li");
      li.textContent = q;
      els.outQuestions.appendChild(li);
    });

    els.analysisSection.classList.add("show");
    els.quickActions.style.display = "block";
    els.chatSection.classList.add("show");
  }

  async function analyzeImage() {
    clearError();
    state.busy = true;
    els.previewWrap.classList.add("show");
    els.previewTitle.innerHTML = 'جارٍ التحليل<span class="spinner"></span>';
    els.previewSub.textContent = "قد يستغرق من ثوانٍ إلى دقيقة حسب جهازك (نموذج محلي)...";
    els.analysisSection.classList.remove("show");
    els.quickActions.style.display = "none";
    els.chatSection.classList.remove("show");
    els.chatLog.innerHTML = "";

    try {
      const payload = await postJson("/api/analyze", {
        image_base64: state.imageBase64,
        mime_type: state.imageMimeType,
      });

      state.history = [
        {
          role: "user",
          content: "حلّل هذه الصورة التعليمية.",
          images: [state.imageBase64],
        },
        { role: "assistant", content: payload.raw_text },
      ];

      els.previewTitle.textContent = "تم التحليل ✓";
      els.previewSub.textContent = TYPE_LABELS[payload.data.content_type] || "تم فهم محتوى الصفحة";
      renderAnalysis(payload.data);
    } catch (err) {
      els.previewTitle.textContent = "تعذّر التحليل";
      els.previewSub.textContent = "حاول رفع الصورة مرة أخرى.";
      showError(err.message || "حدث خطأ غير متوقع أثناء تحليل الصورة.");
    } finally {
      state.busy = false;
    }
  }

  async function handleFile(file) {
    if (!file || !file.type.startsWith("image/")) {
      showError("الرجاء اختيار ملف صورة صالح.");
      return;
    }
    clearError();
    try {
      const { dataUrl, base64, mimeType } = await resizeImageFile(file);
      state.imageBase64 = base64;
      state.imageMimeType = mimeType;
      els.previewImg.src = dataUrl;
      analyzeImage();
    } catch (err) {
      showError(err.message || "تعذّرت معالجة الصورة.");
    }
  }

  function appendMessage(role, htmlOrText, isHtml) {
    const div = document.createElement("div");
    div.className = `msg ${role}`;
    if (isHtml) div.innerHTML = htmlOrText;
    else div.textContent = htmlOrText;
    els.chatLog.appendChild(div);
    div.scrollIntoView({ behavior: "smooth", block: "end" });
    return div;
  }

  async function sendChat(text) {
    if (!text.trim() || state.busy) return;
    clearError();
    state.busy = true;
    els.chatSend.disabled = true;
    appendMessage("user", text);
    state.history.push({ role: "user", content: text });

    const typingEl = appendMessage("assistant typing", "يكتب الآن... (قد يستغرق وقتاً على جهازك)");

    try {
      const payload = await postJson("/api/chat", { history: state.history });
      state.history.push({ role: "assistant", content: payload.text });
      typingEl.classList.remove("typing");
      typingEl.innerHTML = mdToHtml(payload.text);
    } catch (err) {
      typingEl.remove();
      showError(err.message || "تعذّر إرسال الرسالة، حاول مرة أخرى.");
      state.history.pop();
    } finally {
      state.busy = false;
      els.chatSend.disabled = false;
    }
  }

  // Events
  els.cameraBtn.addEventListener("click", () => els.cameraInput.click());
  els.galleryBtn.addEventListener("click", () => els.galleryInput.click());
  els.cameraInput.addEventListener("change", (e) => {
    if (e.target.files[0]) handleFile(e.target.files[0]);
  });
  els.galleryInput.addEventListener("change", (e) => {
    if (e.target.files[0]) handleFile(e.target.files[0]);
  });

  ["dragover", "dragenter"].forEach((evt) => {
    els.dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      els.dropzone.classList.add("dragover");
    });
  });
  ["dragleave", "drop"].forEach((evt) => {
    els.dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      els.dropzone.classList.remove("dragover");
    });
  });
  els.dropzone.addEventListener("drop", (e) => {
    const file = e.dataTransfer.files && e.dataTransfer.files[0];
    if (file) handleFile(file);
  });

  els.resetBtn.addEventListener("click", () => {
    state.imageBase64 = null;
    state.imageMimeType = null;
    state.history = [];
    els.previewWrap.classList.remove("show");
    els.analysisSection.classList.remove("show");
    els.quickActions.style.display = "none";
    els.chatSection.classList.remove("show");
    els.chatLog.innerHTML = "";
    clearError();
    els.cameraInput.value = "";
    els.galleryInput.value = "";
    window.scrollTo({ top: 0, behavior: "smooth" });
  });

  document.querySelectorAll(".quick-chip").forEach((btn) => {
    btn.addEventListener("click", () => sendChat(btn.dataset.q));
  });

  els.chatForm.addEventListener("submit", (e) => {
    e.preventDefault();
    const text = els.chatInput.value;
    els.chatInput.value = "";
    sendChat(text);
  });
})();
