let model;

window.onload = async () => {
  model = await tf.loadLayersModel('tfjs_model/model.json');
  console.log("Model loaded!");
};

document.getElementById("upload").onchange = async (event) => {
  const file = event.target.files[0];
  if (!file || !model) {
    alert("Model not loaded yet or no file!");
    return;
  }

  const img = new Image();
  img.src = URL.createObjectURL(file);

  img.onload = async () => {
    const tensor = preprocessImage(img); // Your preprocessing function
    const prediction = model.predict(tensor);
    const result = await prediction.data();
    console.log(result);
  };
};
