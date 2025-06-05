(async () => {
  // 1)  把模型 URL 设成绝对路径或带仓库前缀
  const MODEL_URL = `${location.pathname}best.onnx`;    // ⭐️

  // 2)  告诉 ORT wasm 文件在哪（同目录）
  ort.env.wasm.wasmPaths = `${location.pathname}`;

  // 3)  加载模型
  const session = await ort.InferenceSession.create(MODEL_URL);
  console.log('✅ ONNX loaded');

  // 4)  文件选择
  const fileInput = document.getElementById('fileInput');
  const canvas    = document.getElementById('canvas');
  const ctx       = canvas.getContext('2d');

  fileInput.onchange = async e => {
    const file = e.target.files[0];
    if (!file) return;

    // 4-1  画图
    const img = new Image();
    img.src   = URL.createObjectURL(file);
    await img.decode();
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

    // 4-2  预处理到 tensor (这里仅示例，按你的模型需要来)
    const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const input   = new ort.Tensor(
      'float32',
      Float32Array.from(imgData.data).map(v => v / 255),
      [1, 3, canvas.height, canvas.width]
    );

    // 4-3  推理
    const { output0 } = await session.run({ images: input });

    // 4-4  TODO: 解析 output0 并在 canvas 上绘框
    console.log(output0);
  };
})();
