const startBtn = document.getElementById("start-btn");
const stopBtn = document.getElementById("stop-btn");
const micIndicator = document.getElementById("mic-indicator");
const statusText = document.getElementById("status-text");
const languageSelect = document.getElementById("language");
const resultSection = document.getElementById("result");
const transcriptEl = document.getElementById("transcript");
const answerEl = document.getElementById("answer");
const player = document.getElementById("player");
const tagline1 = document.getElementById("tagline-1");
const tagline2 = document.getElementById("tagline-2");
const heroSub = document.getElementById("hero-sub");
const historyLink = document.getElementById("history-link");
const logoutLink = document.getElementById("logout-link");

let mediaRecorder;
let chunks = [];
let isRecording = false;

// Localized UI strings for the currently selected language.
// Falls back to sensible English defaults until the first fetch resolves.
let uiStrings = {
  status_default: "Press Start and speak",
  status_listening: "Listening... press Stop when done",
  status_thinking: "Thinking...",
  status_mic_denied: "Microphone access denied or unavailable.",
  status_error: "Something went wrong. Try again.",
};

function setStatus(key) {
  statusText.textContent = uiStrings[key] || uiStrings.status_default;
}

function setRecordingUI(recording) {
  isRecording = recording;
  micIndicator.classList.toggle("recording", recording);
  startBtn.disabled = recording;
  stopBtn.disabled = !recording;
}

function showResult({ transcript, answer_text, audio_url, message }) {
  resultSection.classList.remove("hidden");
  transcriptEl.textContent = transcript ? `"${transcript}"` : "";
  answerEl.textContent = answer_text || message || "";
  if (audio_url) {
    player.src = audio_url;
    player.play().catch(() => {
      /* autoplay may be blocked; user can press play manually */
    });
  } else {
    player.removeAttribute("src");
  }
}

// --- Localization: translate the whole page whenever the language changes ---

async function applyLanguage(language) {
  try {
    const [stringsResp, intentsResp] = await Promise.all([
      fetch(`/api/ui-strings?language=${encodeURIComponent(language)}`),
      fetch(`/api/intents?language=${encodeURIComponent(language)}`),
    ]);
    uiStrings = await stringsResp.json();
    const intents = await intentsResp.json();

    tagline1.textContent = uiStrings.tagline_line1;
    tagline2.textContent = uiStrings.tagline_line2;
    heroSub.textContent = uiStrings.hero_sub;
    historyLink.textContent = uiStrings.history_link;
    logoutLink.textContent = uiStrings.logout_link;
    startBtn.textContent = uiStrings.start_btn;
    stopBtn.textContent = uiStrings.stop_btn;

    if (!isRecording) {
      setStatus("status_default");
    }

    const labelById = {};
    intents.forEach((i) => { labelById[i.id] = i.label; });
    document.querySelectorAll(".chip").forEach((btn) => {
      const label = labelById[btn.dataset.intent];
      if (label) btn.textContent = label;
    });
  } catch (err) {
    console.error("Failed to load translations for", language, err);
  }
}

languageSelect.addEventListener("change", () => {
  applyLanguage(languageSelect.value);
});

// Translate to whatever language is selected by default on page load.
applyLanguage(languageSelect.value);

// --- Recording ---

async function startRecording() {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  chunks = [];
  mediaRecorder = new MediaRecorder(stream);
  mediaRecorder.ondataavailable = (e) => chunks.push(e.data);
  mediaRecorder.onstop = onRecordingStop;
  mediaRecorder.start();
  setRecordingUI(true);
  setStatus("status_listening");
}

function stopRecording() {
  if (mediaRecorder && isRecording) {
    mediaRecorder.stop();
    mediaRecorder.stream.getTracks().forEach((t) => t.stop());
    setRecordingUI(false);
  }
}

async function onRecordingStop() {
  setStatus("status_thinking");
  const blob = new Blob(chunks, { type: "audio/webm" });
  const formData = new FormData();
  formData.append("audio", blob, "query.webm");
  formData.append("language", languageSelect.value);

  try {
    const resp = await fetch("/api/voice-query", { method: "POST", body: formData });
    const data = await resp.json();
    setStatus("status_default");
    showResult(data);
  } catch (err) {
    setStatus("status_error");
    console.error(err);
  }
}

startBtn.addEventListener("click", () => {
  startRecording().catch((err) => {
    setStatus("status_mic_denied");
    console.error(err);
  });
});

stopBtn.addEventListener("click", () => {
  stopRecording();
});

document.querySelectorAll(".chip").forEach((btn) => {
  btn.addEventListener("click", async () => {
    setStatus("status_thinking");
    try {
      const resp = await fetch("/api/text-query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          intent_id: btn.dataset.intent,
          language: languageSelect.value,
        }),
      });
      const data = await resp.json();
      setStatus("status_default");
      showResult(data);
    } catch (err) {
      setStatus("status_error");
      console.error(err);
    }
  });
});
