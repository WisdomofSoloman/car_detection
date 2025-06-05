/* ---------- 全局 ---------- */
const MODEL_URL = "best.onnx";   // 与 index.html 同目录
const CANVAS_SZ = 608;

/* ---------- 加载模型 ---------- */
(async () => {
  await ort.env.setPromisePolyfill();        // 兼容旧浏览器
  ort.env.wasm.wasmPaths = "./";             // wasm 文件就在同目录（CDN 会自动下载）
  const session = await ort.InferenceSession.create(MODEL_URL);
  console.log("✅ ONNX loaded");

  const fileInput = document.getElementById("fileInput");
  const canvas    = document.getElementById("canvas");
  const ctx       = canvas.getContext("2d");

  fileInput.onchange = async ev => {
    const file = ev.target.files[0];
    if(!file) return;

    /* 1. 读入并 letter-box 到 608×608 */
    const img = new Image();
    img.src = URL.createObjectURL(file);
    await img.decode();

    const scale = Math.min(CANVAS_SZ/img.width, CANVAS_SZ/img.height);
    const nw = img.width*scale, nh = img.height*scale;
    const padX = (CANVAS_SZ-nw)/2, padY = (CANVAS_SZ-nh)/2;

    ctx.fillStyle = "rgb(114,114,114)";
    ctx.fillRect(0,0,CANVAS_SZ,CANVAS_SZ);
    ctx.drawImage(img, padX, padY, nw, nh);

    /* 2. 转 tensor(CHW, float32, 0-1) */
    const imgData = ctx.getImageData(0,0,CANVAS_SZ,CANVAS_SZ).data;
    const chw = new Float32Array(3*CANVAS_SZ*CANVAS_SZ);
    for(let i=0,p=0;i<imgData.length;i+=4,++p){
      chw[p]                 = imgData[i]/255;
      chw[p+CANVAS_SZ**2]    = imgData[i+1]/255;
      chw[p+CANVAS_SZ**2*2]  = imgData[i+2]/255;
    }
    const input = new ort.Tensor("float32", chw, [1,3,CANVAS_SZ,CANVAS_SZ]);

    /* 3. 推理（输出已 NMS，shape=[N,6]）*/
    const out = await session.run({images: input});
    const det = out[Object.keys(out)[0]].data;   // Float32Array

    /* 4. 画框 */
    ctx.drawImage(img, padX, padY, nw, nh);      // 重画一次原图，抹掉 letterbox 灰边
    ctx.lineWidth = 2;
    ctx.font = "18px Arial";
    ctx.textBaseline = "top";
    for(let i=0;i<det.length;i+=6){
      const [x1,y1,x2,y2,conf] = det.slice(i,i+5);
      if(conf < 0.35) continue;                 // 前端再过滤一次
      /* 坐标已是 letter-box 空间，直接减灰边再缩放回原图尺寸 */
      const bx1 = (x1 - padX)/scale, by1 = (y1 - padY)/scale;
      const bx2 = (x2 - padX)/scale, by2 = (y2 - padY)/scale;

      ctx.strokeStyle = "lime";
      ctx.fillStyle   = "lime";
      ctx.beginPath();
      ctx.rect(bx1, by1, bx2-bx1, by2-by1);
      ctx.stroke();
      ctx.fillText(`car ${(conf*100).toFixed(1)}%`, bx1+2, by1+2);
    }
  };
})();
