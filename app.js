/* -------------------------------------------------
 * Car-Detector ONNX demo  (xyxy + 已含 NMS 版本)
 * ------------------------------------------------- */
(async () => {
  /* 1. ORT 初始化 */
  ort.env.wasm.wasmPaths = "./";
  ort.env.wasm.simd      = true;          // 浏览器支持则会加载 SIMD 版
  const session = await ort.InferenceSession.create("./best.onnx", {
    executionProviders: ["wasm"]
  });
  console.log("✅  ONNX 模型加载完成");

  /* 2. DOM */
  const input  = document.getElementById("fileInput");
  const cvs    = document.getElementById("canvas");
  const ctx    = cvs.getContext("2d");
  const SIZE   = 608;                     // 模型固定输入

  /* 3. 监听上传并推理 */
  input.onchange = async ev => {
    const file = ev.target.files[0];
    if (!file) return;

    /* 3-1 读取并 Letter-box 到 608×608（保持比例） */
    const img = new Image();
    img.src   = URL.createObjectURL(file);
    await img.decode();

    const scale = Math.min(SIZE / img.width, SIZE / img.height);
    const dw    = (SIZE - img.width  * scale) / 2;
    const dh    = (SIZE - img.height * scale) / 2;

    ctx.clearRect(0, 0, SIZE, SIZE);
    ctx.drawImage(img, dw, dh, img.width * scale, img.height * scale);

    /* 3-2 取像素 → Tensor (CHW, float32, 0-1) */
    const { data } = ctx.getImageData(0, 0, SIZE, SIZE);
    const chw = new Float32Array(3 * SIZE * SIZE);
    for (let i = 0, p = 0; i < data.length; i += 4, p++) {
      const [r, g, b] = [data[i], data[i+1], data[i+2]];
      chw[p]                 = r / 255;
      chw[p +   SIZE*SIZE]   = g / 255;
      chw[p + 2*SIZE*SIZE]   = b / 255;
    }
    const tensor = new ort.Tensor("float32", chw, [1, 3, SIZE, SIZE]);

    /* 3-3 推理（模型已包含 NMS，直接得到最终框） */
    const outputMap = await session.run({ images: tensor });
    const y = outputMap[Object.keys(outputMap)[0]].data;   // [N,6]

    /* 3-4 解析输出 */
    const boxes = [];
    for (let i = 0; i < y.length; i += 6) {
      const score = y[i + 4];
      if (score < 0.25) continue;             // ← 置信度阈值
      boxes.push({
        x1: y[i],       y1: y[i+1],
        x2: y[i+2],     y2: y[i+3],
        score
      });
    }

    /* 3-5 重新绘图（先把背景重新画一次，避免旧框残留） */
    ctx.clearRect(0, 0, SIZE, SIZE);
    ctx.drawImage(img, dw, dh, img.width * scale, img.height * scale);

    /* 3-6 画框 */
    ctx.strokeStyle = "lime";
    ctx.lineWidth   = 2;
    ctx.fillStyle   = "lime";
    ctx.font        = "18px sans-serif";

    boxes.forEach(b => {
      const w = b.x2 - b.x1;
      const h = b.y2 - b.y1;
      ctx.strokeRect(b.x1, b.y1, w, h);
      ctx.fillText(`car ${(b.score*100).toFixed(1)}%`, b.x1 + 3, b.y1 + 18);
    });

    console.log(`detections: ${boxes.length}`);
  };
})();
