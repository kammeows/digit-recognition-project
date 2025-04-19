let model;

window.onload = async () => {
  document.getElementById("upload").disabled = true;
  // model = await tf.loadLayersModel('tfjs_model/model.json');
  model = await tf.loadLayersModel('https://raw.githubusercontent.com/kammeows/digit-recognition-project/master/tfjs_model/model.json');
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
    const canvas = document.createElement('canvas');
    canvas.width = 28;
    canvas.height = 28;
    const ctx = canvas.getContext('2d');

    ctx.drawImage(img, 0, 0, 28, 28);

    let tensor = tf.browser.fromPixels(canvas).toFloat();
    tensor = tensor.div(tf.scalar(255));
    tensor = tensor.expandDims(0);
    tensor = tensor.mean(-1).expandDims(-1);

    const prediction = model.predict(tensor);
    const result = await prediction.data();
    console.log(result);
    document.getElementById("result").textContent = `Prediction: ${result}`;
  };
};
