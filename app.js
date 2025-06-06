/* =============== 全局配置 =============== */
const CFG = {
  MODEL_URL  : "best.onnx",
  INPUT_SIZE : 608,
  STRIDES    : [8, 16, 32],
  SCORE_THRES: 0.55,
  NMS_IOU    : 0.3
};

/* ------------- 工具函数 -------------- */
const sigmoid = x => 1 / (1 + Math.exp(-x));
function iou(a, b) {
  const [ax1, ay1, ax2, ay2] = a, [bx1, by1, bx2, by2] = b;
  const inter =
    Math.max(0, Math.min(ax2, bx2) - Math.max(ax1, bx1)) *
    Math.max(0, Math.min(ay2, by2) - Math.max(ay1, by1));
  const union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter;
  return inter / union;
}
function nms(boxes) {
  boxes.sort((a, b) => b.score - a.score);
  const keep = [];
  while (boxes.length) {
    const box = boxes.shift();
    keep.push(box);
    boxes = boxes.filter(b => iou(box.xyxy, b.xyxy) < CFG.NMS_IOU);
  }
  return keep;
}
/* 生成 (grid_x, grid_y, stride) 列表，与输出顺序相同 */
function buildGrids(size = CFG.INPUT_SIZE) {
  const grids = [];
  CFG.STRIDES.forEach(s => {
    const ny = Math.ceil(size / s);
    const nx = Math.ceil(size / s);
    for (let y = 0; y < ny; ++y)
      for (let x = 0; x < nx; ++x)
        grids.push([x, y, s]);
  });
  return grids;
}

/* ------------- 主逻辑 -------------- */
(async () => {
  /* 1. 载入 onnxruntime-web */
  if (!window.ort) {
    console.error("onnxruntime-web script not loaded!");
    return;
  }
  ort.env.wasm.wasmPaths = "./";
  const session = await ort.InferenceSession.create(CFG.MODEL_URL);
  console.log("✅ ONNX loaded");
  window.session = session;          // 方便调试

  /* 2. DOM */
  const fileInput = document.getElementById("fileInput");
  const showCvs   = document.getElementById("canvas");
  const workCvs   = document.getElementById("work");
  const showCtx   = showCvs.getContext("2d");
  const workCtx   = workCvs.getContext("2d");

  /* 3. 预生成 grids —— 7581 行 */
  const GRIDS = buildGrids(CFG.INPUT_SIZE);

  /* 4. 上传并推理 */
  fileInput.onchange = async e => {
    const file = e.target.files[0];
    if (!file) return;

    /* 4-1 加载图片到显示画布 */
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
    workCtx.drawImage(
      img, 0, 0, img.width, img.height,
      padX, padY, img.width * scale, img.height * scale
    );

    /* 4-3 RGBA → CHW Float32 */
    const rgba = workCtx.getImageData(0, 0, CFG.INPUT_SIZE, CFG.INPUT_SIZE).data;
    const chw  = new Float32Array(3 * CFG.INPUT_SIZE * CFG.INPUT_SIZE);
    for (let i = 0, p = 0; i < rgba.length; i += 4, ++p) {
      chw[p]                       = rgba[i]   / 255;
      chw[p + CFG.INPUT_SIZE**2]   = rgba[i+1] / 255;
      chw[p + 2*CFG.INPUT_SIZE**2] = rgba[i+2] / 255;
    }
    const tensor = new ort.Tensor("float32", chw, [1, 3, CFG.INPUT_SIZE, CFG.INPUT_SIZE]);

    /* 4-4 推理 */
    const t0 = performance.now();
    const out = Object.values(await session.run({ images: tensor }))[0]; // [1,5,N]
    console.log(`⏱ ${(performance.now()-t0).toFixed(1)} ms`);

    /* 4-5 解码 + 置信度过滤 */
    const [ , , N] = out.dims;
    const d = out.data;
    const boxes = [];
    for (let i = 0; i < N; ++i) {
      const [gx, gy, s] = GRIDS[i];
      const cx = ((sigmoid(d[i])       * 2 - 0.5) + gx) * s;
      const cy = ((sigmoid(d[i+  N])   * 2 - 0.5) + gy) * s;
      const w  =  Math.pow(sigmoid(d[i+2*N]) * 2, 2) * s;
      const h  =  Math.pow(sigmoid(d[i+3*N]) * 2, 2) * s;
      const score = sigmoid(d[i+4*N]);
      if (score < CFG.SCORE_THRES) continue;
      boxes.push({ score, xyxy:[cx-w/2, cy-h/2, cx+w/2, cy+h/2] });
    }

    /* 4-6 NMS */
    const keep = Array.isArray(nms(boxes)) ? nms(boxes) : [];
    console.log(`🔍 kept ${keep.length} boxes`);

    /* 4-7 绘制到原图坐标 */
    showCtx.lineWidth   = 2;
    showCtx.strokeStyle = "#00FF00";
    showCtx.fillStyle   = "#00FF00";
    showCtx.font        = "18px Arial";

    keep.forEach(b => {
      let [x1, y1, x2, y2] = b.xyxy;
      x1 = (x1 - padX) / scale;
      y1 = (y1 - padY) / scale;
      x2 = (x2 - padX) / scale;
      y2 = (y2 - padY) / scale;

      const w = x2 - x1, h = y2 - y1;
      showCtx.strokeRect(x1, y1, w, h);
      showCtx.fillText(`car ${(b.score*100).toFixed(1)}%`,
        x1, y1 > 20 ? y1 - 5 : y1 + 20);
    });
  };
})();
