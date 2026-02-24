document.addEventListener("DOMContentLoaded", ()=>{

  console.log("V2 script loaded");

  const mediaUpload = document.getElementById("mediaUpload");
  const preview = document.getElementById("preview");
  const result = document.getElementById("result");
  const loader = document.getElementById("loader");
  const predictBtn = document.getElementById("predictBtn");
  const audioPlayer = document.getElementById("ttsAudio");
  const logoutBtn = document.getElementById("logoutBtn");
  const dropZone = document.getElementById("dropZone");

  /* ================= LOGOUT ================= */
  if(logoutBtn){
    logoutBtn.onclick = ()=> window.location="/logout";
  }

  /* ================= OPEN FILE PICKER ================= */
  if(dropZone){
    dropZone.onclick = ()=> mediaUpload.click();
  }

  /* ================= FILE CHANGE ================= */
  mediaUpload.addEventListener("change", ()=>{
    const file = mediaUpload.files[0];
    if(!file) return;

    const url = URL.createObjectURL(file);
    preview.innerHTML = `<video src="${url}" controls></video>`;
  });

  /* ================= DRAG & DROP ================= */
  if(dropZone){

    dropZone.addEventListener("dragover",(e)=>{
      e.preventDefault();
      dropZone.classList.add("drag");
    });

    dropZone.addEventListener("dragleave",()=>{
      dropZone.classList.remove("drag");
    });

    dropZone.addEventListener("drop",(e)=>{
      e.preventDefault();
      dropZone.classList.remove("drag");

      const file = e.dataTransfer.files[0];
      if(file){
        mediaUpload.files = e.dataTransfer.files;
        const url = URL.createObjectURL(file);
        preview.innerHTML = `<video src="${url}" controls></video>`;
      }
    });
  }

  /* ================= PREDICT ================= */
  if(predictBtn){
    predictBtn.onclick = async ()=>{

      const file = mediaUpload.files[0];
      if(!file) return alert("Upload video first");

      loader.classList.remove("hidden");
      result.innerText = "...";

      try{
        const form = new FormData();
        form.append("file", file);

        const res = await fetch("/predict_video_v2",{
          method:"POST",
          body:form
        });

        if(!res.ok){
          const txt = await res.text();
          throw new Error(txt);
        }

        const data = await res.json();

        result.innerText = (data.sentences || []).join(" ");

        if(data.audio && data.audio[0]){
          audioPlayer.src = data.audio[0];
          audioPlayer.play();
        }

      }catch(err){
        console.error(err);
        alert("Prediction failed — check backend");
      }

      loader.classList.add("hidden");
    };
  }

});