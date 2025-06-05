/******************************************************************************
 *  app.js - 纯前端 ONNXRuntime-Web 推理 + 结果绘制
 ******************************************************************************/

(async () => {
  // -------------------------------------------------------------
  // 0)  确认 ORT 已加载 & 指定 wasm 文件路径
  // -------------------------------------------------------------
  if (!window.ort) {
    console.error('❌ onnxruntime-web script not loaded!');
    return;
  }
  // 让 ORT 在当前目录下寻找 *.wasm
  ort.env.wasm.wasmPaths = location.pathname;

  // -------------------------------------------------------------
  // 1)  加载模型
  // -------------------------------------------------------------
  const MODEL_URL = `${location.pathname}best.onnx`; // 同目录下
  let session;
  try {
    session = await ort.InferenceSession.create(MODEL_URL);
    console.log('✅ ONNX model loaded');
  } catch (e) {
    console.error('❌ Failed to load model:', e);
    return;
  }

  // -------------------------------------------------------------
  // 2)  绑定 DOM
  // -------------------------------------------------------------
  const fileInput = document.getElementById('fileInput');
  const canvas    = document.getElementById('canvas');
  const ctx       = canvas.getContext('2d');

  fileInput.addEventListener('change', handleFile, false);

  // -------------------------------------------------------------
  // 3)  处理上传文件
  // -------------------------------------------------------------
  async function handleFile (ev) {
    const file = ev.target.files[0];
    if (!file) return;

    // 3-1) 在 <canvas> 上画出原图
    const img = new Image();
    img.src   = URL.createObjectURL(file);
    await img.decode();
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

    // 3-2) 读取像素，转换为 (1,3,H,W) float32 tensor，简单归一化到 [0,1]
    const imgData   = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
    const chwData   = new Float32Array(3 * canvas.height * canvas.width);
    for (let i = 0, p = 0; i < imgData.length; i += 4, ++p) {
      const [r, g, b] = [imgData[i], imgData[i + 1], imgData[i + 2]];
      chwData[p              ] = r / 255;                 // R
      chwData[p +  canvas.w * canvas.h] = g / 255;        // G
      chwData[p + 2*canvas.w * canvas.h] = b / 255;       // B
    }
    const inputTensor = new ort.Tensor('float32', chwData,
                                       [1, 3, canvas.height, canvas.width]);

    // 3-3) 推理
    let output;
    try {
      output = await session.run({ images: inputTensor });
    } catch (e) {
      console.error('❌ Inference failed:', e);
      return;
    }
    const out = output[Object.keys(output)[0]]; // 取第一个输出

    // ---------------------------------------------------------
    // 4) 解析 YOLOv8 输出并绘制（只支持 1 类 car 的简化版）
    //    out shape = (1, 5, n): [cx,cy,w,h,conf]
    // ---------------------------------------------------------
    drawBoxes(out.data, /*scoreThr*/0.3);
  }

  /*********************  utils  *************************/
  function drawBoxes (flat, scoreThr = 0.3) {
    const stride = 5;               // 每个检测 5 个数字
    ctx.lineWidth   = 2;
    ctx.strokeStyle = '#00FF00';
    ctx.font        = '16px monospace';
    ctx.fillStyle   = '#00FF00';

    for (let i = 0; i < flat.length; i += stride) {
      const [cx, cy, w, h, conf] = flat.slice(i, i + stride);
      if (conf < scoreThr) continue;

      const x = (cx - w / 2) * canvas.width;
      const y = (cy - h / 2) * canvas.height;
      const bw = w * canvas.width;
      const bh = h * canvas.height;

      ctx.strokeRect(x, y, bw, bh);
      ctx.fillText(`car ${(conf*100).toFixed(1)}%`, x + 3, y + 16);
    }
  }
})();
