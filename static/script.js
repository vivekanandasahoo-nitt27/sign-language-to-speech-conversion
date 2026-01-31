let useWebcam = false;
let lastPrediction = "";
let webcamInterval = null;

// Elements
const uploadTab = document.getElementById("uploadTab");
const liveTab = document.getElementById("liveTab");
const uploadSection = document.getElementById("uploadSection");
const liveSection = document.getElementById("liveSection");
const dropZone = document.getElementById("dropZone");
const mediaUpload = document.getElementById("mediaUpload");
const preview = document.getElementById("preview");
const webcam = document.getElementById("webcam");
const predictBtn = document.getElementById("predictBtn");
const loader = document.getElementById("loader");
const resultBox = document.getElementById("result");
const audioPlayer = document.getElementById("ttsAudio");

// ===== Tabs =====
uploadTab.addEventListener("click", () => {
  useWebcam = false;
  uploadSection.classList.remove("hidden");
  liveSection.classList.add("hidden");
  uploadTab.classList.add("active");
  liveTab.classList.remove("active");

  stopLivePrediction();
  mediaUpload.click();
});

liveTab.addEventListener("click", () => {
  useWebcam = true;
  uploadSection.classList.add("hidden");
  liveSection.classList.remove("hidden");
  liveTab.classList.add("active");
  uploadTab.classList.remove("active");

  startWebcam();
});

// ===== Drag & Drop + Click Upload =====
dropZone.addEventListener("click", () => mediaUpload.click());

dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropZone.style.background = "#e0f2fe";
});

dropZone.addEventListener("dragleave", () => {
  dropZone.style.background = "#f9fafb";
});

dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.style.background = "#f9fafb";
  handleMedia(e.dataTransfer.files[0]);
});

mediaUpload.addEventListener("change", function () {
  handleMedia(this.files[0]);
});

// ===== Handle Image or Video =====
function handleMedia(file) {
  if (!file) return;

  preview.innerHTML = "";
  resultBox.innerText = "---";
  lastPrediction = "";

  const url = URL.createObjectURL(file);

  if (file.type.startsWith("image/")) {
    const img = document.createElement("img");
    img.src = url;
    preview.appendChild(img);
  } 
  else if (file.type.startsWith("video/")) {
    const video = document.createElement("video");
    video.src = url;
    video.controls = true;
    preview.appendChild(video);
  }
}

// ===== Webcam =====
async function startWebcam() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    webcam.srcObject = stream;
  } catch (err) {
    alert("⚠️ Webcam access denied!");
  }
}

// ===== Predict Button =====
predictBtn.addEventListener("click", async () => {
  loader.classList.remove("hidden");
  

  try {
    const formData = new FormData();

    if (useWebcam) {
      stopLivePrediction();
      loader.classList.add("hidden");
      startLivePrediction();
      return;
    }

    const file = mediaUpload.files[0];
    if (!file) {
      alert("Please upload a file first!");
      loader.classList.add("hidden");
      return;
    }

    formData.append("file", file);

    const endpoint = file.type.startsWith("image/")
      ? "/predict_image"
      : "/predict_video";

    const response = await fetch(endpoint, {
      method: "POST",
      body: formData,
    });

    const data = await response.json();  // ✅ SAFE


    // 🔥 FIX: Handle both response styles safely
    const resultText =
      data.prediction ||
      data.sentence ||
      data.label ||
      data.error ||
      "No result";

    const previousText = lastPrediction;
    typeText(resultText);

    // 🔊 Autoplay only if prediction changed
    if (data.audio && resultText !== previousText) {
      try {
        audioPlayer.pause();
        audioPlayer.src = data.audio;
        audioPlayer.currentTime = 0;
        audioPlayer.play();
      } catch (audioErr) {
        console.warn("Audio play failed:", audioErr);
      }
    }

  } catch (err) {
    console.error("❌ Network / JS Error:", err);
    alert("⚠️ Frontend error. Check console.");
  }

  loader.classList.add("hidden");
});

// ===== Live webcam prediction loop =====
function startLivePrediction() {
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");

  webcamInterval = setInterval(async () => {
    if (!useWebcam) return stopLivePrediction();

    if (webcam.videoWidth === 0) return;

    canvas.width = webcam.videoWidth;
    canvas.height = webcam.videoHeight;
    ctx.drawImage(webcam, 0, 0);

    const blob = await new Promise((resolve) =>
      canvas.toBlob(resolve, "image/jpeg")
    );

    const formData = new FormData();
    formData.append("file", blob, "frame.jpg");

    try {
      const response = await fetch("/predict_image", {
        method: "POST",
        body: formData,
      });

      const result = await response.json();
      const text =
        result.prediction ||
        result.label ||
        "No result";

      typeText(text);

    } catch (err) {
      console.error(err);
    }
  }, 700); // smoother than 500ms
}

function stopLivePrediction() {
  if (webcamInterval) {
    clearInterval(webcamInterval);
    webcamInterval = null;
  }
}

// ===== Update result text =====
function typeText(text) {
  if (!text || text === lastPrediction) return;
  lastPrediction = text;
  resultBox.innerText = text;
}
