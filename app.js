// 1. 载入模型
let session;
(async () => {
  session = await ort.InferenceSession.create('./best.onnx');
  console.log('ONNX model loaded');
})();

// 2. 监听文件选择
document.getElementById('fileInput').addEventListener('change', async (e) => {
  const file   = e.target.files[0];
  if (!file || !session) return;

  // 2-1 读图并画在 canvas
  const imgURL = URL.createObjectURL(file);
  const img    = new Image();
  img.src      = imgURL;
  await img.decode();

  const cvs = document.getElementById('canvas');
  const ctx = cvs.getContext('2d');
  ctx.drawImage(img, 0, 0, cvs.width, cvs.height);

  // 2-2 把图片转成 NCHW Float32       (这里只做演示，未做归一化 / 608×608 resize)
  const tensor = new ort.Tensor('float32', new Float32Array(cvs.width * cvs.height * 3), [1,3,cvs.height,cvs.width]);
  // … 实际项目请用 GPU 预处理或 wasm SIMD …

  // 2-3 推理
  const feeds = { images: tensor };   // 名字与模型输入一致
  const res   = await session.run(feeds);
  document.getElementById('log').textContent = JSON.stringify(res, null, 2);
});
