/* ----------------  配置  ---------------- */
const MODEL_URL   = "best.onnx";     // 放在根目录即可
const INPUT_SIZE  = 608;             // 训练 / 导出时用的尺寸
const SCORE_THRES = 0.30;            // 置信度阈值
const NMS_IOU     = 0.45;            // NMS IoU 阈值

/* ------------- 工具函数 -------------- */
const sigmoid = x => 1 / (1 + Math.exp(-x));

function iou(boxA, boxB) {
  const [ax1, ay1, ax2, ay2] = boxA;
  const [bx1, by1, bx2, by2] = boxB;
  const interArea =
    Math.max(0, Math.min(ax2, bx2) - Math.max(ax1, bx1)) *
    Math.max(0, Math.min(ay2, by2) - Math.max(ay1, by1));
  const unionArea =
    (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - interArea;
  return interArea / unionArea;
}

function nms(boxes) {
  boxes.sort((a, b) => b.score - a.score);
  const keep = [];
  while (boxes.length) {
    const box = boxes.shift();
    keep.push(box);
    boxes = boxes.filter(b => iou(box.xyxy, b.xyxy) < NMS_IOU);
  }
  return keep;
}

/* ------------- 主逻辑 -------------- */
(async () => {
  /* 1. 载入 ort */
  if (!window.ort) {
    console.error("onnxruntime-web script not loaded!");
    return;
  }
  ort.env.wasm.wasmPaths = "./";           // 告诉它 wasm 放在同目录
  const session = await ort.InferenceSession.create(MODEL_URL);
  console.log("✅ ONNX loaded");

  /* 2. DOM */
  const fileInput = document.getElementById("fileInput");
  const showCvs   = document.getElementById("canvas");
  const workCvs   = document.getElementById("work");
  const showCtx   = showCvs.getContext("2d");
  const workCtx   = workCvs.getContext("2d");

  /* 3. 处理上传 */
  fileInput.onchange = async e => {
    const file = e.target.files[0];
    if (!file) return;

    /* 3-1 读图 & 按比例绘制到展示画布 */
    const img = new Image();
    img.src = URL.createObjectURL(file);
    await img.decode();

    // 让展示画布跟图一样比例
    showCvs.width  = img.width;
    showCvs.height = img.height;
    showCtx.drawImage(img, 0, 0);

    /* 3-2 letter-box => 608×608，得到在原图中的 scale/offset  */
    let scale = Math.min(INPUT_SIZE / img.width, INPUT_SIZE / img.height);
    let padX  = (INPUT_SIZE - img.width  * scale) / 2;
    let padY  = (INPUT_SIZE - img.height * scale) / 2;

    workCtx.fillStyle = "rgba(114,114,114,1)";   // 与 Ultralytics 一致
    workCtx.fillRect(0, 0, INPUT_SIZE, INPUT_SIZE);
    workCtx.drawImage(
      img,
      0, 0, img.width, img.height,
      padX, padY, img.width * scale, img.height * scale
    );

    /* 3-3 取像素 -> Float32[1,3,608,608] */
    const imgData = workCtx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE).data;
    const chw = new Float32Array(3 * INPUT_SIZE * INPUT_SIZE);
    for (let i = 0, p = 0; i < imgData.length; i += 4, ++p) {
      chw[p]                       = imgData[i]   / 255;       // R
      chw[p + INPUT_SIZE**2]       = imgData[i+1] / 255;       // G
      chw[p + 2*INPUT_SIZE**2]     = imgData[i+2] / 255;       // B
    }
    const tensor = new ort.Tensor("float32", chw, [1, 3, INPUT_SIZE, INPUT_SIZE]);

    /* 3-4  推理 */
    const { output0 } = await session.run({ images: tensor });
    const data = output0.data;          // Float32Array  len = 1×5×7581

    /* 3-5 解析 + NMS */
    const boxes = [];
    const n = data.length / 5;
    for (let i = 0; i < n; ++i) {
      const off = i * 5;
      const [x, y, w, h, obj] = data.slice(off, off + 5).map(sigmoid);

      const score = obj;        // 只有 1 类 = car
      if (score < SCORE_THRES) continue;

      // xywh → xyxy (相对 0-1)
      const cx = x, cy = y;
      const bw = w, bh = h;
      const x1 = (cx - bw / 2);
      const y1 = (cy - bh / 2);
      const x2 = (cx + bw / 2);
      const y2 = (cy + bh / 2);

      boxes.push({ score, xyxy: [x1, y1, x2, y2] });
    }
    const keep = boxes;  

    /* 3-6 绘制结果 (映射回原图坐标) */
    showCtx.lineWidth = 2;
    showCtx.font = "18px Arial";
    showCtx.strokeStyle = "#00FF00";
    showCtx.fillStyle   = "#00FF00";

    keep.forEach(b => {
     let [x1, y1, x2, y2] = b.xyxy;
     x1 = (x1 - padX) / scale;
     y1 = (y1 - padY) / scale;
     x2 = (x2 - padX) / scale;
     y2 = (y2 - padY) / scale;

     const w = x2 - x1, h = y2 - y1;
     showCtx.strokeRect(x1, y1, w, h);
     showCtx.fillText(`car ${(b.score * 100).toFixed(1)}%`, x1, y1 > 20 ? y1 - 5 : y1 + 20);
});


    console.log(`🔍 kept ${keep.length} boxes`);
  };
})();
