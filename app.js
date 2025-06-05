import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/esm/ort.min.js";

const modelURL = "best.onnx";
const inputSize = 608;               // 训练时的 imgsz

// 1. 载入模型（WASM 后端，若浏览器支持则切换到 WebGPU）
const session = await ort.InferenceSession.create(
  modelURL,
  { executionProviders: (await ort.env.wasm.wasmSimd()).length ? ['wasm'] : ['cpu'] }
);

const fileInput = document.getElementById("file");
fileInput.onchange = async (e) => {
  const img = new Image();
  img.src = URL.createObjectURL(e.target.files[0]);
  img.onload = () => run(img);
};

async function run(img) {
  // 2. 预处理 → Float32Array [1,3,608,608]
  const {tensor, ratio, pad} = preprocess(img);

  // 3. 推理
  const outputs = await session.run({images: tensor});
  const [boxes, scores] = postprocess(outputs, ratio, pad);   // 解析 + NMS

  // 4. 渲染
  draw(img, boxes, scores);
}

function preprocess(img) {
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = inputSize;
  const ctx = canvas.getContext('2d');
  // letterbox
  const ratio = Math.min(inputSize/img.width, inputSize/img.height);
  const newW = img.width * ratio, newH = img.height * ratio;
  const dx = (inputSize - newW) / 2, dy = (inputSize - newH) / 2;
  ctx.fillStyle = "#808080"; ctx.fillRect(0,0,inputSize,inputSize);
  ctx.drawImage(img, dx, dy, newW, newH);

  // 取像素并归一化
  const imgData = ctx.getImageData(0,0,inputSize,inputSize).data;
  const float32 = new Float32Array(inputSize*inputSize*3);
  for (let i=0;i<imgData.length;i+=4){
     const j = i/4;                              // NHWC
     float32[j+      0] = imgData[i  ]/255;      // R
     float32[j+inputSize*inputSize] = imgData[i+1]/255; // G
     float32[j+2*inputSize*inputSize] = imgData[i+2]/255; // B
  }
  const tensor = new ort.Tensor("float32", float32, [1,3,inputSize,inputSize]);
  return {tensor, ratio, pad:[dx,dy]};
}

function postprocess(outputMap, ratio, pad){
  // Ultralytics onnx 导出：一个输出节点，shape [1, 5, 7581]
  const data = outputMap[Object.keys(outputMap)[0]].data;
  const boxes = [], scores = [];
  for (let i=0;i<7581;i++){
     const conf = data[4*7581 + i];        // objectness
     if (conf < 0.25) continue;            // conf 阈值
     const x = (data[i*4+0]-pad[0])/ratio;
     const y = (data[i*4+1]-pad[1])/ratio;
     const w = data[i*4+2]/ratio;
     const h = data[i*4+3]/ratio;
     boxes.push([x-w/2,y-h/2,w,h]);        // xywh → xyxy
     scores.push(conf);
  }
  // 简单 NMS (可换 Tensor-NMS)
  return [boxes, scores];
}

function draw(img, boxes, scores){
  const cvs = document.getElementById("canvas");
  cvs.width = img.width; cvs.height = img.height;
  const ctx = cvs.getContext('2d');
  ctx.drawImage(img, 0,0);
  ctx.strokeStyle = "#00ff00"; ctx.lineWidth = 2; ctx.font="16px sans-serif";
  boxes.forEach((b,i)=>{
     ctx.strokeRect(b[0],b[1],b[2],b[3]);
     ctx.fillText(`${(scores[i]*100).toFixed(1)}%`, b[0], b[1]-5);
  });
}
