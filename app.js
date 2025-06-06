/* ---------- 全局配置 ---------- */
const CFG = {
  MODEL_URL   : "best.onnx",   // dynamic=True 导出的 raw logits 模型
  INPUT_SIZE  : 608,
  STRIDES     : [8, 16, 32],
  SCORE_THRES : 0.25,
  NMS_IOU     : 0.45
};

/* ---------- 工具函数 ---------- */
const sigmoid = x => 1 / (1 + Math.exp(-x));
function iou(a, b) { /* 与之前相同 */ }
function nms(boxes) { /* 与之前相同 */ }

/* 生成 (grid_x, grid_y, stride) —— 只需一次 */
function buildGrids(size) {
  const g = [];
  CFG.STRIDES.forEach(s => {
    const ny = Math.ceil(size / s);
    const nx = Math.ceil(size / s);
    for (let y = 0; y < ny; ++y)
      for (let x = 0; x < nx; ++x)
        g.push([x, y, s]);
  });
  return g;
}

(async () => {
  /* 1. 载入 ort */
  if (!window.ort) return console.error("💥 ort.min.js not loaded");
  ort.env.wasm.wasmPaths = "./";
  const session = await ort.InferenceSession.create(CFG.MODEL_URL);
  window.session = session;               // 方便调试

  /* 2. DOM */
  const fileInput = document.getElementById("fileInput");
  const showCvs   = document.getElementById("canvas");
  const workCvs   = document.getElementById("work");
  const showCtx   = showCvs.getContext("2d");
  const workCtx   = workCvs.getContext("2d");

  /* 3. 预生成 grids 对应 7581 个预测点 */
  const GRIDS = buildGrids(CFG.INPUT_SIZE);

  /* 4. 处理上传 */
  fileInput.onchange = async e => {
    const file = e.target.files[0];
    if (!file) return;

    /* 4-1 读图并绘制到展示画布 */
    const img = new Image();
    img.src = URL.createObjectURL(file);
    await img.decode();

    showCvs.width  = img.width;
    showCvs.height = img.height;
    showCtx.clearRect(0, 0, img.width, img.height);
    showCtx.drawImage(img, 0, 0);

    /* 4-2 letter-box → 608×608 */
    const scale = Math.min(CFG.INPUT_SIZE / img.width, CFG.INPUT_SIZE / img.height);
    const padX  = (CFG.INPUT_SIZE - img.width  * scale) / 2;
    const padY  = (CFG.INPUT_SIZE - img.height * scale) / 2;

    workCtx.fillStyle = "rgba(114,114,114,1)";
    workCtx.fillRect(0, 0, CFG.INPUT_SIZE, CFG.INPUT_SIZE);
    workCtx.drawImage(img, 0, 0, img.width, img.height,
                      padX, padY, img.width * scale, img.height * scale);

    /* 4-3 RGBA → Float32 CHW */
    const imgData = workCtx.getImageData(0, 0, CFG.INPUT_SIZE, CFG.INPUT_SIZE).data;
    const chw = new Float32Array(3 * CFG.INPUT_SIZE * CFG.INPUT_SIZE);
    for (let i = 0, p = 0; i < imgData.length; i += 4, ++p) {
      chw[p]                       = imgData[i]   / 255;
      chw[p + CFG.INPUT_SIZE**2]   = imgData[i+1] / 255;
      chw[p + 2*CFG.INPUT_SIZE**2] = imgData[i+2] / 255;
    }
    const tensor = new ort.Tensor("float32", chw, [1, 3, CFG.INPUT_SIZE, CFG.INPUT_SIZE]);

    /* 4-4 推理 */
    const out   = Object.values(await session.run({images: tensor}))[0];  // [1,5,7581]
    const [ , , n] = out.dims;
    const data  = out.data;

    /* 4-5 解码 + 置信度过滤 */
    const boxes = [];
    for (let i = 0; i < n; ++i) {
      const [gx, gy, s] = GRIDS[i];
      const cx = ((sigmoid(data[i])        * 2 - 0.5) + gx) * s;
      const cy = ((sigmoid(data[i +   n])  * 2 - 0.5) + gy) * s;
      const w  =  Math.pow(sigmoid(data[i + 2*n]) * 2, 2) * s;
      const h  =  Math.pow(sigmoid(data[i + 3*n]) * 2, 2) * s;
      const score = sigmoid(data[i + 4*n]);
      if (score < CFG.SCORE_THRES) continue;
      boxes.push({ score, xyxy: [cx - w/2, cy - h/2, cx + w/2, cy + h/2] });
    }

    /* 4-6 NMS & 映射回原图 */
    const keep = nms(boxes);
    showCtx.lineWidth = 2;
    showCtx.strokeStyle = "#00FF00";
    showCtx.fillStyle   = "#00FF00";
    showCtx.font = "18px Arial";

    for (const b of keep) {
      let [x1, y1, x2, y2] = b.xyxy;
      x1 = (x1 - padX) / scale;  y1 = (y1 - padY) / scale;
      x2 = (x2 - padX) / scale;  y2 = (y2 - padY) / scale;
      showCtx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      showCtx.fillText(`car ${(b.score*100).toFixed(1)}%`,
                       x1, y1 > 20 ? y1 - 5 : y1 + 20);
    }
  };
})();
