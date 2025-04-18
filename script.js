let model;

async function loadModel() {
  model = await tf.loadLayersModel("tfjs_model/model.json");
  console.log("Model loaded");
}
loadModel();

document.getElementById("file-input").addEventListener("change", async (e) => {
  const file = e.target.files[0];
  const img = new Image();
  const canvas = document.getElementById("canvas");
  const ctx = canvas.getContext("2d");

  img.onload = async () => {
    ctx.drawImage(img, 0, 0, 28, 28);
    const imageData = ctx.getImageData(0, 0, 28, 28);
    let input = tf.browser
      .fromPixels(imageData, 1)
      .reshape([1, 28, 28, 1])
      .div(255.0);

    const prediction = model.predict(input);
    const predClass = (await prediction.argMax(1).data())[0];
    document.getElementById("result").innerText = `Prediction: ${predClass}`;
  };

  img.src = URL.createObjectURL(file);
});
