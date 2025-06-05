/* -------------------------------------------------
 * Car detector – 前端推理 demo（基本版）
 * 目录结构假设：
 *   ├─ index.html
 *   ├─ app.js          ←  就是本文件
 *   ├─ best.onnx
 *   ├─ ort.min.js
 *   ├─ ort-wasm.wasm
 *   └─ ort-wasm-simd.wasm   (可选，但建议一起放)
 * ------------------------------------------------- */

(async () => {
  /* 1️⃣ 告诉 ORT：wasm 文件在哪（就在当前目录） */
  ort.env.wasm.wasmPaths = "./";

  /* 2️⃣ 若浏览器支持 SIMD，则自动加载 ort-wasm-simd.wasm */
  ort.env.wasm.simd = true;

  /* 3️⃣ 加载模型（只用 wasm 后端最稳） */
  const session = await ort.InferenceSession.create("./best.onnx", {
    executionProviders: ["wasm"]
  });
  console.log("✅ ONNX model ready");

  /* 4️⃣ DOM */
  const fileInput = document.getElementById("fileInput");
  const canvas    = document.getElementById("canvas");
  const ctx       = canvas.getContext("2d");
  const SIZE      = 608;          // 模型输入分辨率

  /* 5️⃣ 监听上传 */
  fileInput.addEventListener("change", async e => {
    const file = e.target.files[0];
    if (!file) return;

    /* 5-1 读图并等解码完成 */
    const img = new Image();
    img.src   = URL.createObjectURL(file);
    await img.decode();

    /* 5-2 Letterbox 到 608×608（不保持宽高比最简单）*/
    ctx.drawImage(img, 0, 0, SIZE, SIZE);

    /* 5-3 取像素 → CHW Float32(0-1) */
    const { data } = ctx.getImageData(0, 0, SIZE, SIZE);
    const chw = new Float32Array(3 * SIZE * SIZE);
    for (let i = 0, p = 0; i < data.length; i += 4, p++) {
      const [r, g, b] = [data[i], data[i + 1], data[i + 2]];
      chw[p]                      = r / 255;           // R
      chw[p +   SIZE * SIZE]      = g / 255;           // G
      chw[p + 2*SIZE * SIZE]      = b / 255;           // B
    }
    const tensor = new ort.Tensor("float32", chw, [1, 3, SIZE, SIZE]);

    /* 5-4 推理 */
    const outputMap = await session.run({ images: tensor });
    const output    = outputMap[Object.keys(outputMap)[0]]; // 第一张输出

    /* 5-5 解析 – 这里只给出最简 mAP-0.5 NMS，假设输出为 [N, 6]：
       [x1, y1, x2, y2, score, class]，且只训练了 'car' = 0
       *你的导出的 YOLOv8 onnx 默认就是这种格式*
    */
    const raw = output.data;
    const boxes = [];
    for (let i = 0; i < raw.length; i += 6) {
      const score = raw[i + 4];
      if (score < 0.25) continue;              // 置信度阈值
      boxes.push({
        x1: raw[i]     * SIZE,
        y1: raw[i + 1] * SIZE,
        x2: raw[i + 2] * SIZE,
        y2: raw[i + 3] * SIZE,
        score
      });
    }

    /* 简单 NMS（IOU 0.45） */
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

    /* 5-6 重新把原图按等比缩放画到画布中央（可选，保持纵横比）*/
    const scale = Math.min(SIZE / img.width, SIZE / img.height);
    const dw = (SIZE - img.width  * scale) / 2;
    const dh = (SIZE - img.height * scale) / 2;
    ctx.clearRect(0, 0, SIZE, SIZE);
    ctx.drawImage(img, dw, dh, img.width * scale, img.height * scale);

    /* 5-7 画框 */
    ctx.strokeStyle = "lime";
    ctx.lineWidth   = 2;
    ctx.font        = "16px sans-serif";
    ctx.fillStyle   = "lime";

    kept.forEach(b => {
      const x = b.x1, y = b.y1, w = b.x2 - b.x1, h = b.y2 - b.y1;
      ctx.strokeRect(x, y, w, h);
      ctx.fillText(`car ${(b.score*100).toFixed(1)}%`, x+2, y+16);
    });
    console.log(`detections: ${kept.length}`);
  });

})();
