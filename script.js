let model;

window.onload = async () => {
  document.getElementById("upload").disabled = true;
  model = await tf.loadLayersModel('tfjs_model/model.json');
  console.log("Model loaded!");
  document.getElementById("upload").disabled = false;
};

document.getElementById("upload").onchange = async (event) => {
  if (!model) {
    alert("Model not loaded yet!");
    return;
  }
  const file = event.target.files[0];
  if (!file) {
    alert("No file selected!");
    return;
  }

  const img = new Image();
  img.src = URL.createObjectURL(file);

  img.onload = async () => {
    const tensor = preprocessImage(img);
    if (!tensor) {
      alert("Preprocessing failed!");
      return;
    }
    const prediction = model.predict(tensor);
    const result = await prediction.data();
    console.log(result);
  };
};
