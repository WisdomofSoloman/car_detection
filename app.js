/* ----------------  配置  ---------------- */
const MODEL_URL   = "best.onnx";
const SCORE_THRES = 0.30;   // 保留阈值
const NMS_IOU     = 0.45;

/* ---------- NMS & 工具函数 ----------- */
const iou = (a,b)=>{
  const [x1,y1,x2,y2]=a, [x1_,y1_,x2_,y2_]=b;
  const inter = Math.max(0,Math.min(x2,x2_)-Math.max(x1,x1_))
              * Math.max(0,Math.min(y2,y2_)-Math.max(y1,y1_));
  const union = (x2-x1)*(y2-y1)+(x2_-x1_)*(y2_-y1_)-inter;
  return inter/union;
};
const nms = (boxes)=>{
  boxes.sort((a,b)=>b.score-a.score);
  const keep=[];
  while(boxes.length){
    const cur=boxes.shift(); keep.push(cur);
    boxes=boxes.filter(b=>iou(cur.xyxy,b.xyxy)<NMS_IOU);
  }
  return keep;
};

/* -------------- 主流程 --------------- */
(async()=>{
  /* 1. ort wasm */
  if(!window.ort){console.error("🏷 ort 未加载");return;}
  ort.env.wasm.wasmPaths="./";
  const session=await ort.InferenceSession.create(MODEL_URL);
  console.log("✅ onnx ready");

  /* 2. DOM */
  const f   = document.getElementById("fileInput");
  const cv  = document.getElementById("canvas");
  const ctx = cv.getContext("2d");

  f.onchange=async e=>{
    const file=e.target.files[0];
    if(!file) return;
    /* 2-1 画原图 */
    const img=new Image();
    img.src=URL.createObjectURL(file);
    await img.decode();
    cv.width=img.width; cv.height=img.height;
    ctx.drawImage(img,0,0);

    /* 2-2 将整张图缩放到 640×640（YOLOv8 默认导出尺寸）喂模型 */
    const sz   = 640;            // 导出时用的 imgsz
    const tmp  = document.createElement("canvas");
    tmp.width  = tmp.height = sz;
    const tctx = tmp.getContext("2d");
    // letter-box：灰边填充
    const scale = Math.min(sz/img.width, sz/img.height);
    const nw    = img.width*scale;
    const nh    = img.height*scale;
    const dx    = (sz-nw)/2, dy=(sz-nh)/2;
    tctx.fillStyle="rgba(114,114,114,1)";
    tctx.fillRect(0,0,sz,sz);
    tctx.drawImage(img,dx,dy,nw,nh);

    /* 2-3 提取像素 -> [1,3,640,640] */
    const data = tctx.getImageData(0,0,sz,sz).data;
    const chw  = new Float32Array(3*sz*sz);
    for(let i=0,p=0;i<data.length;i+=4,++p){
      chw[p]           = data[i]  /255;   // R
      chw[p+sz*sz]     = data[i+1]/255;   // G
      chw[p+2*sz*sz]   = data[i+2]/255;   // B
    }
    const input=new ort.Tensor("float32",chw,[1,3,sz,sz]);

    /* 2-4 推理 (输出 shape: [1, dets, 6]) */
    const out = await session.run({images:input});
    const pred= out[Object.keys(out)[0]].data; // Float32Array

    /* 2-5 解析 & NMS */
    const dets = [];
    for(let i=0;i<pred.length;i+=6){
      const [x1,y1,x2,y2,score,cls]=pred.slice(i,i+6);
      if(score<SCORE_THRES) continue;
      dets.push({score, xyxy:[x1,y1,x2,y2]});
    }
    const keep = nms(dets);

    /* 2-6 画框 (映射回原图坐标) */
    ctx.lineWidth=2; ctx.font="18px Arial";
    ctx.strokeStyle="lime"; ctx.fillStyle="lime";
    keep.forEach(b=>{
      let [x1,y1,x2,y2]=b.xyxy;
      // 反 letter-box: 先减 pad, 再除 scale
      x1=(x1-dx)/scale; y1=(y1-dy)/scale;
      x2=(x2-dx)/scale; y2=(y2-dy)/scale;
      const w=x2-x1, h=y2-y1;
      ctx.strokeRect(x1,y1,w,h);
      const txt=`car ${(b.score*100).toFixed(1)}%`;
      ctx.fillText(txt,x1,y1>20?y1-5:y1+20);
    });
    console.log(`🎯 detections=${keep.length}`);
  };
})();
