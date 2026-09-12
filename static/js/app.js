const startBtn = document.getElementById("start-btn");
const stopBtn = document.getElementById("stop-btn");
const micIndicator = document.getElementById("mic-indicator");
const statusText = document.getElementById("status-text");
const languageSelect = document.getElementById("language");
const resultSection = document.getElementById("result");
const transcriptEl = document.getElementById("transcript");
const answerEl = document.getElementById("answer");
const player = document.getElementById("player");

let mediaRecorder;
let chunks = [];
let isRecording = false;

function setStatus(text) {
  statusText.textContent = text;
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

async function startRecording() {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  chunks = [];
  mediaRecorder = new MediaRecorder(stream);
  mediaRecorder.ondataavailable = (e) => chunks.push(e.data);
  mediaRecorder.onstop = onRecordingStop;
  mediaRecorder.start();
  setRecordingUI(true);
  setStatus("Listening... press Stop when done");
}

function stopRecording() {
  if (mediaRecorder && isRecording) {
    mediaRecorder.stop();
    mediaRecorder.stream.getTracks().forEach((t) => t.stop());
    setRecordingUI(false);
  }
}

async function onRecordingStop() {
  setStatus("Thinking...");
  const blob = new Blob(chunks, { type: "audio/webm" });
  const formData = new FormData();
  formData.append("audio", blob, "query.webm");
  formData.append("language", languageSelect.value);

  try {
    const resp = await fetch("/api/voice-query", { method: "POST", body: formData });
    const data = await resp.json();
    setStatus("Press Start and speak");
    showResult(data);
  } catch (err) {
    setStatus("Something went wrong. Try again.");
    console.error(err);
  }
}

startBtn.addEventListener("click", () => {
  startRecording().catch((err) => {
    setStatus("Microphone access denied or unavailable.");
    console.error(err);
  });
});

stopBtn.addEventListener("click", () => {
  stopRecording();
});

document.querySelectorAll(".chip").forEach((btn) => {
  btn.addEventListener("click", async () => {
    setStatus("Thinking...");
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
      setStatus("Press Start and speak");
      showResult(data);
    } catch (err) {
      setStatus("Something went wrong. Try again.");
      console.error(err);
    }
  });
});
