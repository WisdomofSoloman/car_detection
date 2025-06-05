/* -------------------------------------------------
 * Car-Detector ONNX demo  (fixed bbox)
 * ------------------------------------------------- */
(async () => {
  /* ORT 初始化 */
  ort.env.wasm.wasmPaths = "./";
  ort.env.wasm.simd      = true;
  const session = await ort.InferenceSession.create("./best.onnx", {
    executionProviders: ["wasm"]
  });
  console.log("✅ ONNX model ready");

  /* DOM */
  const input  = document.getElementById("fileInput");
  const cvs    = document.getElementById("canvas");
  const ctx    = cvs.getContext("2d");
  const SIZE   = 608;

  /* 上传并推理 */
  input.onchange = async ev => {
    const file = ev.target.files[0];
    if (!file) return;

    /* -------- 1. 读取 & Letter-box -------- */
    const img = new Image();
    img.src   = URL.createObjectURL(file);
    await img.decode();

    const scale = Math.min(SIZE / img.width, SIZE / img.height);
    const dw    = (SIZE - img.width  * scale) / 2;
    const dh    = (SIZE - img.height * scale) / 2;
    ctx.clearRect(0, 0, SIZE, SIZE);
    ctx.drawImage(img, dw, dh, img.width * scale, img.height * scale);

    /* -------- 2. 取像素 → Tensor -------- */
    const { data } = ctx.getImageData(0, 0, SIZE, SIZE);
    const chw = new Float32Array(3 * SIZE * SIZE);
    for (let i = 0, p = 0; i < data.length; i += 4, p++) {
      const [r, g, b] = [data[i], data[i+1], data[i+2]];
      chw[p]                 = r/255;
      chw[p+SIZE*SIZE]       = g/255;
      chw[p+2*SIZE*SIZE]     = b/255;
    }
    const tensor = new ort.Tensor("float32", chw, [1,3,SIZE,SIZE]);

    /* -------- 3. 推理 -------- */
    const out = (await session.run({images: tensor}))[0].data;  // [N,6]

    /* -------- 4. 解析 + 坐标转换 -------- */
    let boxes = [];
    for (let i = 0; i < out.length; i += 6) {
      const score = out[i+4];
      if (score < 0.25) continue;              // 置信度阈值
      // 原始输出是比例中心点格式
      const cx = out[i]     * SIZE;
      const cy = out[i+1]   * SIZE;
      const w  = out[i+2]   * SIZE;
      const h  = out[i+3]   * SIZE;
      const x1 = cx - w/2,
            y1 = cy - h/2,
            x2 = cx + w/2,
            y2 = cy + h/2;
      // 映射回 Letter-box 后的实际像素
      boxes.push({
        x1: x1 * scale + dw,
        y1: y1 * scale + dh,
        x2: x2 * scale + dw,
        y2: y2 * scale + dh,
        score
      });
    }

    /* -------- 5. NMS (IOU 0.45) -------- */
    boxes.sort((a,b)=>b.score-a.score);
    const keep=[], iou=(a,b)=>{
      const xx1=Math.max(a.x1,b.x1), yy1=Math.max(a.y1,b.y1);
      const xx2=Math.min(a.x2,b.x2), yy2=Math.min(a.y2,b.y2);
      const w=Math.max(0,xx2-xx1),   h=Math.max(0,yy2-yy1);
      const inter=w*h, areaA=(a.x2-a.x1)*(a.y2-a.y1),
            areaB=(b.x2-b.x1)*(b.y2-b.y1);
      return inter/(areaA+areaB-inter);
    };
    while(boxes.length){
      const b=boxes.shift(); keep.push(b);
      boxes=boxes.filter(x=>iou(b,x)<0.45);
    }

    /* -------- 6. 重新绘制框 -------- */
    ctx.strokeStyle="lime"; ctx.lineWidth=2;
    ctx.fillStyle="lime";   ctx.font="18px sans-serif";
    keep.forEach(b=>{
      const w=b.x2-b.x1, h=b.y2-b.y1;
      ctx.strokeRect(b.x1,b.y1,w,h);
      ctx.fillText(`car ${(b.score*100).toFixed(1)}%`, b.x1+3, b.y1+18);
    });
    console.log(`detections: ${keep.length}`);
  };
})();
