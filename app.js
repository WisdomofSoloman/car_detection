/* -------------------------------------------------
 * Car-Detector ONNX demo  (xyxy + NMS 版本 · 键名自适应)
 * ------------------------------------------------- */
(async () => {
  try {
    /* 1. ORT 初始化 */
    ort.env.wasm.wasmPaths = "./";
    ort.env.wasm.simd      = true;

    const session = await ort.InferenceSession.create("./best.onnx", {
      executionProviders: ["wasm"]
    });
    console.log("✅  ONNX 模型加载完成");

    /* 1-2 记录真实的输入/输出名 */
    const inName  = session.inputNames[0];   // 例如 'input0' / 'images'
    const outName = session.outputNames[0];  // 例如 'output0'
    console.log(`ℹ️  input  name: ${inName}`);
    console.log(`ℹ️  output name: ${outName}`);

    /* 2. DOM */
    const fileInput = document.getElementById("fileInput");
    const canvas    = document.getElementById("canvas");
    const ctx       = canvas.getContext("2d");
    const SIZE      = 608;                   // fixed input size

    /* 3. 上传并推理 */
    fileInput.addEventListener("change", async ev => {
      const file = ev.target.files[0];
      if (!file) return;

      try {
        /* 3-1 读图片并 letter-box 到 608×608 */
        const img   = new Image();
        img.src     = URL.createObjectURL(file);
        await img.decode();

        const scale = Math.min(SIZE / img.width, SIZE / img.height);
        const dw    = (SIZE - img.width  * scale) / 2;
        const dh    = (SIZE - img.height * scale) / 2;

        ctx.clearRect(0, 0, SIZE, SIZE);
        ctx.drawImage(img, dw, dh, img.width * scale, img.height * scale);

        /* 3-2 CHW Tensor */
        const { data } = ctx.getImageData(0, 0, SIZE, SIZE);
        const chw = new Float32Array(3 * SIZE * SIZE);
        for (let i = 0, p = 0; i < data.length; i += 4, p++) {
          chw[p]                 = data[i]     / 255;
          chw[p +   SIZE*SIZE]   = data[i + 1] / 255;
          chw[p + 2*SIZE*SIZE]   = data[i + 2] / 255;
        }
        const tensor = new ort.Tensor("float32", chw, [1, 3, SIZE, SIZE]);

        /* 3-3 推理 */
        const outputMap = await session.run({ [inName]: tensor });
        const y = outputMap[outName];              // Tensor 对象
        if (!y) {
          console.error("❌ 推理成功，但没拿到输出！", outputMap);
          return;
        }

        /* 3-4 解析 xyxy + conf */
        const boxes = [];
        const raw = y.data;                        // TypedArray
        for (let i = 0; i < raw.length; i += 6) {
          const score = raw[i + 4];
          if (score < 0.25) continue;
          boxes.push({
            x1: raw[i],  y1: raw[i + 1],
            x2: raw[i + 2], y2: raw[i + 3],
            score
          });
        }

        /* 3-5 重绘背景再画框 */
        ctx.clearRect(0, 0, SIZE, SIZE);
        ctx.drawImage(img, dw, dh, img.width * scale, img.height * scale);

        ctx.strokeStyle = "lime";
        ctx.lineWidth   = 2;
        ctx.fillStyle   = "lime";
        ctx.font        = "18px sans-serif";
        boxes.forEach(b => {
          ctx.strokeRect(b.x1, b.y1, b.x2 - b.x1, b.y2 - b.y1);
          ctx.fillText(`car ${(b.score*100).toFixed(1)}%`, b.x1 + 3, b.y1 + 18);
        });

        console.log(`✅  detections : ${boxes.length}`);
      } catch (err) {
        console.error("❌  推理或绘制时出错：", err);
      }
    });

  } catch (err) {
    console.error("❌  模型加载失败：", err);
  }
})();
