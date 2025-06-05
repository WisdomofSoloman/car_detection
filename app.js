/* -------------------------------------------------
 * Car detector – 前端推理 demo（修订版）
 * ------------------------------------------------- */

(async () => {
  /* 1️⃣ ORT wasm 文件路径（当前目录） */
  ort.env.wasm.wasmPaths = "./";
  ort.env.wasm.simd      = true;  // 有 SIMD 就用 SIMD

  /* 2️⃣ 加载 ONNX（仅 wasm 后端） */
  const session = await ort.InferenceSession.create("./best.onnx", {
    executionProviders: ["wasm"]
  });
  console.log("✅ ONNX model ready");

  /* 3️⃣ DOM */
  const fileInput = document.getElementById("fileInput");
  const canvas    = document.getElementById("canvas");
  const ctx       = canvas.getContext("2d");
  const SIZE      = 608;          // 模型输入分辨率

  /* 4️⃣ 监听上传 */
  fileInput.addEventListener("change", async e => {
    const file = e.target.files[0];
    if (!file) return;

    /* ---- 4-1 读取图片 ---- */
    const img = new Image();
    img.src   = URL.createObjectURL(file);
    await img.decode();

    /* ---- 4-2 Letter-box 到 608×608 ---- */
    const scale = Math.min(SIZE / img.width, SIZE / img.height);
    const dw    = (SIZE - img.width  * scale) / 2;
    const dh    = (SIZE - img.height * scale) / 2;
    ctx.clearRect(0, 0, SIZE, SIZE);
    ctx.drawImage(img, dw, dh, img.width * scale, img.height * scale);

    /* ---- 4-3 像素数据 → Tensor(CHW 0-1) ---- */
    const { data } = ctx.getImageData(0, 0, SIZE, SIZE);
    const chw = new Float32Array(3 * SIZE * SIZE);
    for (let i = 0, p = 0; i < data.length; i += 4, p++) {
      const [r, g, b] = [data[i], data[i + 1], data[i + 2]];
      chw[p]                      = r / 255;              // R
      chw[p +   SIZE * SIZE]      = g / 255;              // G
      chw[p + 2*SIZE * SIZE]      = b / 255;              // B
    }
    const tensor = new ort.Tensor("float32", chw, [1, 3, SIZE, SIZE]);

    /* ---- 4-4 运行推理 ---- */
    const outputMap = await session.run({ images: tensor });
    const output    = outputMap[Object.keys(outputMap)[0]].data; // [N,6]

    /* ---- 4-5 解析 + 置信度阈值 ---- */
    let boxes = [];                                   // ← 用 let！
    for (let i = 0; i < output.length; i += 6) {
      const score = output[i + 4];
      if (score < 0.25) continue;                     // 置信度过滤
      boxes.push({
        x1: output[i]     ,      // 0-608 坐标
        y1: output[i + 1] ,
        x2: output[i + 2] ,
        y2: output[i + 3] ,
        score
      });
    }

    /* ---- 4-6 NMS (IOU 0.45) ---- */
    boxes.sort((a, b) => b.score - a.score);
    const kept = [];
    const iou = (a, b) => {
      const xx1 = Math.max(a.x1, b.x1);
      const yy1 = Math.max(a.y1, b.y1);
      const xx2 = Math.min(a.x2, b.x2);
      const yy2 = Math.min(a.y2, b.y2);
      const w   = Math.max(0, xx2 - xx1);
      const h   = Math.max(0, yy2 - yy1);
      const inter = w * h;
      const union = (a.x2 - a.x1) * (a.y2 - a.y1) +
                    (b.x2 - b.x1) * (b.y2 - b.y1) - inter;
      return inter / union;
    };
    while (boxes.length) {
      const box = boxes.shift();
      kept.push(box);
      boxes = boxes.filter(b => iou(box, b) < 0.45);
    }

    /* ---- 4-7 重新绘制（图已在 4-2 画好） ---- */
    ctx.strokeStyle = "lime";
    ctx.lineWidth   = 2;
    ctx.font        = "18px sans-serif";
    ctx.fillStyle   = "lime";

    kept.forEach(b => {
      // 把 0-608 坐标还原到 Letter-box 后的实际像素
      const x = b.x1 * scale + dw;
      const y = b.y1 * scale + dh;
      const w = (b.x2 - b.x1) * scale;
      const h = (b.y2 - b.y1) * scale;

      ctx.strokeRect(x, y, w, h);
      ctx.fillText(`car ${(b.score*100).toFixed(1)}%`, x + 4, y + 18);
    });

    console.log(`detections (kept): ${kept.length}`);
  });
})();
