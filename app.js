/* ----------------  全局配置  ---------------- */
const MODEL_URL   = "best.onnx";  // 放在根目录
const SCORE_THRES = 0.25;
const IOU_THRES   = 0.45;

/* ---------- IoU & NMS ---------- */
function boxIou(a, b) {
  const [x1,y1,x2,y2] = a, [x1_,y1_,x2_,y2_] = b;
  const inter = Math.max(0, Math.min(x2,x2_)-Math.max(x1,x1_)) *
                Math.max(0, Math.min(y2,y2_)-Math.max(y1,y1_));
  const union = (x2-x1)*(y2-y1)+(x2_-x1_)*(y2_-y1_)-inter;
  return inter / union;
}
function nms(boxes) {
  boxes.sort((a,b)=>b.conf-a.conf);
  const keep=[];
  while (boxes.length) {
    const cur = boxes.shift();
    keep.push(cur);
    boxes = boxes.filter(b=>boxIou(cur.xyxy, b.xyxy) < IOU_THRES);
  }
  return keep;
}

/* -------------- 推理流程 -------------- */
(async () => {
  /* 载入 ort wasm & 模型 */
  ort.env.wasm.wasmPaths = "./";              // wasm 文件与 html 同目录
  const session = await ort.InferenceSession.create(MODEL_URL);
  console.log("✅ ONNX loaded");

  /* DOM */
  const inp = document.getElementById("fileInput");
  const cvs = document.getElementById("canvas");
  const ctx = cvs.getContext("2d");

  inp.onchange = async ev => {
    const file = ev.target.files[0];
    if (!file) return;

    /* 1. 显示原图 */
    const img = new Image();
    img.src   = URL.createObjectURL(file);
    await img.decode();
    cvs.width  = img.width;
    cvs.height = img.height;
    ctx.drawImage(img, 0, 0);

    /* 2. letter-box 到 640×640  */
    const SZ   = 640;
    const tmp  = document.createElement("canvas");
    tmp.width  = tmp.height = SZ;
    const tctx = tmp.getContext("2d");
    const scale = Math.min(SZ/img.width, SZ/img.height);
    const nw = img.width*scale,  nh = img.height*scale;
    const padX = (SZ - nw) / 2, padY = (SZ - nh) / 2;
    tctx.fillStyle = "rgb(114,114,114)";
    tctx.fillRect(0,0,SZ,SZ);
    tctx.drawImage(img, padX, padY, nw, nh);

    /* 3. HWC → CHW → Float32 */
    const data = tctx.getImageData(0,0,SZ,SZ).data;
    const chw  = new Float32Array(3*SZ*SZ);
    for (let i=0,p=0;i<data.length;i+=4,++p){
      chw[p]          = data[i]   /255;      // R
      chw[p+SZ*SZ]    = data[i+1]/255;      // G
      chw[p+2*SZ*SZ]  = data[i+2]/255;      // B
    }
    const input = new ort.Tensor("float32", chw, [1,3,SZ,SZ]);

    /* 4. 推理 */
    const out = await session.run({images: input});
    const raw = out[Object.keys(out)[0]].data;  // Float32Array, [N,6]

    /* 5. 解析 -> xyxy */
    const boxes=[];
    for (let i=0;i<raw.length;i+=6){
      const cx=raw[i],  cy=raw[i+1];
      const w =raw[i+2], h =raw[i+3];
      const conf = raw[i+4];
      if (conf < SCORE_THRES) continue;

      // 5-1 center → corner (仍在 640×640 坐标系)
      let x1=cx-w/2, y1=cy-h/2, x2=cx+w/2, y2=cy+h/2;

      // 5-2 映射回原图
      x1 = (x1 - padX) / scale;
      y1 = (y1 - padY) / scale;
      x2 = (x2 - padX) / scale;
      y2 = (y2 - padY) / scale;

      // 裁剪到画布
      x1=Math.max(0, x1); y1=Math.max(0,y1);
      x2=Math.min(img.width, x2); y2=Math.min(img.height, y2);

      boxes.push({conf, xyxy:[x1,y1,x2,y2]});
    }

    const dets = nms(boxes);

    /* 6. 绘框 */
    ctx.lineWidth=2;
    ctx.font="18px Arial";
    ctx.textBaseline="top";
    dets.forEach(b=>{
      const [x1,y1,x2,y2]=b.xyxy;
      ctx.strokeStyle="lime";
      ctx.fillStyle="lime";
      ctx.beginPath();
      ctx.rect(x1,y1,x2-x1,y2-y1);
      ctx.stroke();
      ctx.fillText(`car ${(b.conf*100).toFixed(1)}%`, x1+2, y1+2);
    });
    console.log(`🎯 kept ${dets.length} boxes`);
  };
})();
