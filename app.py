"""
app.py — Web App kiểm tra chất lượng trái cây
==============================================
Chạy: python app.py
Mở : http://localhost:5000

Cài đặt: pip install flask opencv-python scikit-image scikit-learn numpy joblib pillow
"""

import os, time, io, base64, traceback
import cv2, numpy as np, joblib
from flask import Flask, request, jsonify, render_template_string
from PIL import Image
from skimage.feature import hog as skimage_hog, local_binary_pattern

# ── Cấu hình ──────────────────────────────────────────────────────────────────
MODEL_PATH = './models/fruit_svm.pkl'
IMG_SIZE   = (224, 224)

FRUITS    = ['Apple','Banana','Guava','Lime','Orange']
QUALITIES = ['Good','Bad','Mixed']
ALL_CLASSES  = sorted([f'{fr}_{q}' for fr in FRUITS for q in QUALITIES])
CLASS_TO_IDX = {c:i for i,c in enumerate(ALL_CLASSES)}
IDX_TO_CLASS = {i:c for c,i in CLASS_TO_IDX.items()}

FRUIT_EMOJI = {'Apple':'🍎','Banana':'🍌','Guava':'🍈','Lime':'🍋','Orange':'🍊'}
QUALITY_COLORS = {'Good':'#27ae60','Bad':'#e74c3c','Mixed':'#f39c12'}

# ── Pipeline xử lý ảnh (giống notebook) ───────────────────────────────────────
def load_resize(img_bgr):
    return cv2.resize(img_bgr, IMG_SIZE, interpolation=cv2.INTER_AREA)

def preprocess(img):
    denoised = cv2.GaussianBlur(img, (5,5), 1.0)
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    lab[:,:,0] = cv2.createCLAHE(2.0,(8,8)).apply(lab[:,:,0])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return enhanced, cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)

def segment_grabcut(img, margin=0.07, iters=5):
    h,w = img.shape[:2]; mx,my = int(w*margin),int(h*margin)
    rect = (mx,my,w-2*mx,h-2*my)
    mask_gc = np.zeros((h,w),np.uint8)
    bgd = np.zeros((1,65),np.float64); fgd = np.zeros((1,65),np.float64)
    cv2.grabCut(img, mask_gc, rect, bgd, fgd, iters, cv2.GC_INIT_WITH_RECT)
    return np.where((mask_gc==cv2.GC_FGD)|(mask_gc==cv2.GC_PR_FGD),255,0).astype(np.uint8)

def detect_defects(img, mask):
    working = cv2.bitwise_and(img,img,mask=mask)
    gray    = cv2.cvtColor(working,cv2.COLOR_BGR2GRAY)
    blur    = cv2.GaussianBlur(gray,(7,7),0)
    _,thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    thresh  = cv2.bitwise_and(thresh,thresh,mask=mask)
    kern    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kern,iterations=2)
    sure_bg = cv2.dilate(opening,kern,iterations=3)
    dist    = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    _,sure_fg = cv2.threshold(dist,0.4*dist.max(),255,0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)
    _,markers = cv2.connectedComponents(sure_fg)
    markers+=1; markers[unknown==255]=0
    cv2.watershed(img.copy(),markers)
    defect = np.zeros(img.shape[:2],np.uint8); defect[markers==-1]=255
    defect = cv2.dilate(defect,kern,iterations=1)
    defect = cv2.bitwise_and(defect,defect,mask=mask)
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    dark = cv2.inRange(hsv,np.array([0,0,0]),np.array([180,80,80]))
    dark = cv2.morphologyEx(cv2.bitwise_and(dark,dark,mask=mask),cv2.MORPH_CLOSE,kern)
    combined = cv2.bitwise_or(defect,dark)
    ratio = int(combined.sum()/255)/max(int(mask.sum()/255),1)
    return combined, float(ratio)

def extract_hog(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    feat,_ = skimage_hog(gray,orientations=9,pixels_per_cell=(16,16),
                          cells_per_block=(2,2),block_norm='L2-Hys',
                          visualize=True,feature_vector=True)
    return feat.astype(np.float32)

def extract_lbp(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    lbp  = local_binary_pattern(gray,24,3,method='uniform')
    hist,_ = np.histogram(lbp.ravel(),bins=64,range=(0,26))
    hist = hist.astype(np.float32); hist/=(hist.sum()+1e-6)
    return hist

def extract_features(img, mask=None, defect_ratio=0.0):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    hog_f = extract_hog(img)
    lbp_f = extract_lbp(img)
    # Color hist HSV
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    ch  = []
    for c,(lo,hi) in enumerate([(0,180),(0,256),(0,256)]):
        h = cv2.calcHist([hsv],[c],mask,[32],[lo,hi]).flatten().astype(np.float32)
        h/=(h.sum()+1e-6); ch.append(h)
    color_hist = np.concatenate(ch)
    # Color stats
    stats = []
    for conv in [None,cv2.COLOR_BGR2HSV,cv2.COLOR_BGR2LAB]:
        sp = cv2.cvtColor(img,conv) if conv else img
        for c in range(3):
            px = sp[:,:,c].astype(np.float32)
            px = px[mask>0] if mask is not None else px.flatten()
            stats += [px.mean() if len(px)>0 else 0, px.std() if len(px)>0 else 0]
    color_stats = np.array(stats,dtype=np.float32)
    # Shape
    total = img.shape[0]*img.shape[1]
    fm = mask if mask is not None else np.ones(img.shape[:2],np.uint8)*255
    cnts,_ = cv2.findContours(fm,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        cnt = max(cnts,key=cv2.contourArea)
        a = cv2.contourArea(cnt); p = cv2.arcLength(cnt,True)
        x,y,w,h = cv2.boundingRect(cnt)
        ha = cv2.contourArea(cv2.convexHull(cnt))
        circ = (4*np.pi*a/p**2) if p>0 else 0
        extent=a/max(w*h,1); solidity=a/max(ha,1); aspect=w/max(h,1)
        hu = -np.sign(cv2.HuMoments(cv2.moments(cnt)).flatten())*\
              np.log10(np.abs(cv2.HuMoments(cv2.moments(cnt)).flatten())+1e-10)
        shape = np.array([a/total,circ,extent,solidity,aspect,*hu],dtype=np.float32)
    else:
        shape = np.zeros(12,dtype=np.float32)
    # Edge
    canny = cv2.Canny(gray,50,150)
    sx = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=3)
    sy = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=3)
    mag = np.clip(np.sqrt(sx**2+sy**2),0,255).astype(np.uint8)
    edge = np.array([float(np.sum(canny>0)/canny.size),float(mag.mean()),float(np.sum(canny>0))],dtype=np.float32)
    return np.concatenate([hog_f,lbp_f,color_hist,color_stats,shape,edge,np.array([defect_ratio],dtype=np.float32)])

def render_vis(img, mask_fruit, mask_defect, pred_label, confidence, defect_ratio):
    vis = img.copy()
    cnts_f,_ = cv2.findContours(mask_fruit, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts_d,_ = cv2.findContours(mask_defect,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis,cnts_f,-1,(0,220,0),2)
    cv2.drawContours(vis,cnts_d,-1,(0,0,220),2)
    ov = vis.copy(); ov[mask_defect>0]=(0,0,180)
    vis = cv2.addWeighted(ov,0.3,vis,0.7,0)
    q   = pred_label.split('_')[1] if '_' in pred_label else '?'
    bc  = {'Good':(34,139,34),'Bad':(0,0,200),'Mixed':(0,140,255)}.get(q,(100,100,100))
    cv2.rectangle(vis,(0,0),(IMG_SIZE[0],44),bc,-1)
    cv2.putText(vis,f'  {pred_label}',(6,30),cv2.FONT_HERSHEY_SIMPLEX,0.85,(255,255,255),2)
    iy = IMG_SIZE[1]-64
    cv2.rectangle(vis,(0,iy-4),(IMG_SIZE[0],IMG_SIZE[1]),(25,25,25),-1)
    for i,txt in enumerate([f'Defect:{defect_ratio*100:.1f}%  Conf:{confidence*100:.0f}%',
                              f'Quality: {q}']):
        cv2.putText(vis,txt,(8,iy+18+i*22),cv2.FONT_HERSHEY_SIMPLEX,0.5,(230,230,230),1)
    return vis

def img_to_b64(img_bgr):
    pil = Image.fromarray(cv2.cvtColor(img_bgr,cv2.COLOR_BGR2RGB))
    buf = io.BytesIO(); pil.save(buf,format='PNG')
    return base64.b64encode(buf.getvalue()).decode()

# ── Flask ──────────────────────────────────────────────────────────────────────
app   = Flask(__name__)
MODEL = None

def get_model():
    global MODEL
    if MODEL is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f'Model không tìm thấy: {MODEL_PATH}\n'
                'Hãy chạy 02_FruitQuality_Pipeline.ipynb trước!'
            )
        MODEL = joblib.load(MODEL_PATH)
    return MODEL

# ── HTML ───────────────────────────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>🍎 Fruit Quality Inspector</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
:root{
  --g1:#1B5E20;--g2:#2E7D32;--g3:#E8F5E9;
  --red:#e74c3c;--orange:#f39c12;--blue:#2980b9;
  --bg:#F0F4F8;--card:#fff;--border:#dde;
  --shadow:0 2px 12px rgba(0,0,0,.10);--r:12px;
}
body{font-family:'Segoe UI',Arial,sans-serif;background:var(--bg);color:#1a1a1a;min-height:100vh}

/* Header */
header{background:linear-gradient(135deg,var(--g1),var(--g2));color:#fff;
       padding:26px 32px;text-align:center;box-shadow:0 4px 16px rgba(0,0,0,.2)}
header h1{font-size:1.9em;letter-spacing:.5px}
header p{opacity:.88;margin-top:5px;font-size:.93em}

/* Layout */
.wrap{max-width:1120px;margin:28px auto;padding:0 18px;
      display:grid;grid-template-columns:1fr 1fr;gap:22px}
@media(max-width:800px){.wrap{grid-template-columns:1fr}}
.full{grid-column:1/-1}

/* Card */
.card{background:var(--card);border-radius:var(--r);box-shadow:var(--shadow);padding:26px}
.card h2{font-size:1.05em;color:var(--g2);margin-bottom:16px;
          border-bottom:2px solid var(--g3);padding-bottom:7px}

/* Drop zone */
#drop{border:2.5px dashed var(--border);border-radius:var(--r);padding:40px 20px;
      text-align:center;cursor:pointer;transition:.2s;background:var(--bg)}
#drop:hover,#drop.over{border-color:var(--g2);background:var(--g3)}
#drop .ico{font-size:2.8em}
#drop p{color:#666;margin-top:8px;font-size:.88em}
#file-in{display:none}
#prev-wrap{margin-top:14px;text-align:center;display:none}
#prev-img{max-width:100%;max-height:230px;border-radius:8px;border:1px solid var(--border)}
#file-name{margin-top:6px;font-size:.82em;color:#666}

/* Buttons */
.btn{display:inline-flex;align-items:center;gap:7px;padding:10px 24px;
     border-radius:8px;font-size:.93em;font-weight:600;cursor:pointer;border:none;transition:.2s}
.btn-p{background:var(--g2);color:#fff}.btn-p:hover{background:var(--g1);transform:translateY(-1px)}
.btn-p:disabled{opacity:.45;cursor:not-allowed;transform:none}
.btn-s{background:#eee;color:#555}.btn-s:hover{background:#ddd}
.btn-row{display:flex;gap:10px;margin-top:16px;flex-wrap:wrap}

/* Spinner */
.spin-wrap{display:none;text-align:center;margin:18px 0}
.spinner{display:inline-block;width:34px;height:34px;border:4px solid #ddd;
          border-top-color:var(--g2);border-radius:50%;animation:sp .8s linear infinite}
@keyframes sp{to{transform:rotate(360deg)}}

/* Result */
#empty{text-align:center;color:#90A4AE;padding:55px 0}
#empty .ico{font-size:2.8em}

.badge{display:inline-flex;align-items:center;gap:9px;
       padding:9px 20px;border-radius:50px;font-size:1.08em;font-weight:700;margin-bottom:16px}
.badge-good{background:#E8F5E9;color:#1B5E20;border:2px solid var(--g2)}
.badge-bad {background:#FFEBEE;color:#B71C1C;border:2px solid var(--red)}
.badge-mixed{background:#FFF3E0;color:#BF360C;border:2px solid var(--orange)}

.metrics{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:14px}
.mbox{background:var(--bg);border-radius:8px;padding:12px 14px;border-left:4px solid var(--g2)}
.mbox.warn{border-left-color:var(--orange)}
.mbox label{font-size:.75em;color:#666;display:block;text-transform:uppercase;letter-spacing:.5px}
.mbox span{font-size:1.25em;font-weight:700}

.top3 li{display:flex;justify-content:space-between;align-items:center;
          padding:6px 0;border-bottom:1px solid var(--border);font-size:.88em}
.top3 li:last-child{border:none}
.pbar{height:6px;background:var(--g2);border-radius:3px;display:inline-block;min-width:3px}

.img-row{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:12px}
.img-row img{width:100%;border-radius:8px;border:1px solid var(--border)}
.img-row figcaption{text-align:center;font-size:.78em;color:#666;margin-top:3px}

/* History */
#hist-list{max-height:300px;overflow-y:auto}
.hi{display:flex;align-items:center;gap:11px;padding:9px 6px;
    border-bottom:1px solid var(--border);cursor:pointer;border-radius:6px;transition:.15s}
.hi:hover{background:var(--g3)}
.hi img{width:46px;height:46px;object-fit:cover;border-radius:6px}
.hi .inf strong{font-size:.88em}
.hi .inf small{color:#666;font-size:.76em;display:block}

/* Progress bar */
#prog-wrap{display:none;margin-top:10px}
#prog-bar{height:6px;background:var(--g2);border-radius:3px;transition:width .4s;width:0}

footer{text-align:center;color:#90A4AE;padding:20px;font-size:.82em}
</style>
</head>
<body>
<header>
  <h1>🍎 Fruit Quality Inspection System</h1>
  <p>Hệ thống Phân loại & Kiểm tra Chất lượng Trái Cây — Xử lý Ảnh & Thị Giác Máy Tính (121036)</p>
</header>

<div class="wrap">
  <!-- Upload -->
  <div class="card">
    <h2>📤 Tải ảnh lên</h2>
    <div id="drop" onclick="document.getElementById('file-in').click()">
      <div class="ico">🖼️</div>
      <strong>Kéo & thả ảnh vào đây</strong>
      <p>hoặc click để chọn · JPG PNG BMP · Tối đa 15MB</p>
    </div>
    <input type="file" id="file-in" accept="image/*">
    <div id="prev-wrap">
      <img id="prev-img" src="" alt="preview">
      <p id="file-name"></p>
    </div>
    <div id="prog-wrap"><div id="prog-bar"></div></div>
    <div class="btn-row">
      <button class="btn btn-p" id="btn-analyze" onclick="analyze()" disabled>🔍 Phân tích</button>
      <button class="btn btn-s" onclick="reset_()">↺ Xóa</button>
    </div>
    <div class="spin-wrap" id="spinner"><div class="spinner"></div><p style="margin-top:8px;color:#666;font-size:.85em">Đang phân tích...</p></div>

    <div style="margin-top:18px;padding:13px;background:var(--g3);border-radius:8px;font-size:.82em;line-height:1.7">
      <strong>📌 Pipeline</strong><br>
      <b>Ch.2</b> Resize 224×224 → Gaussian → CLAHE<br>
      <b>Ch.4</b> GrabCut → Watershed → Defect ratio<br>
      <b>Ch.3</b> HOG(16px) + LBP + Màu HSV + Shape<br>
      <b>Ch.5</b> PCA(200D) → LinearSVC → 15 lớp
    </div>
  </div>

  <!-- Result -->
  <div class="card">
    <h2>📊 Kết quả</h2>
    <div id="empty">
      <div class="ico">🍊</div>
      <p style="margin-top:10px">Tải ảnh và nhấn <b>Phân tích</b> để xem kết quả.</p>
    </div>
    <div id="result-section" style="display:none">
      <div id="badge" class="badge"></div>
      <div class="metrics">
        <div class="mbox"><label>🍎 Loại trái cây</label><span id="r-fruit">—</span></div>
        <div class="mbox"><label>⭐ Chất lượng</label><span id="r-quality">—</span></div>
        <div class="mbox"><label>📊 Độ tin cậy</label><span id="r-conf">—</span></div>
        <div class="mbox warn"><label>🔍 Tỷ lệ khuyết tật</label><span id="r-defect">—</span></div>
        <div class="mbox" style="grid-column:1/-1"><label>⏱️ Thời gian xử lý</label><span id="r-time">—</span></div>
      </div>
      <div style="margin-bottom:12px">
        <strong style="font-size:.88em;color:#666">Top-3 dự đoán:</strong>
        <ul class="top3" id="top3" style="margin-top:7px;list-style:none"></ul>
      </div>
      <div class="img-row" id="img-row">
        <figure><img id="img-orig" src="" alt="Original"><figcaption>Ảnh gốc (224×224)</figcaption></figure>
        <figure><img id="img-res"  src="" alt="Result"><figcaption>Kết quả phân đoạn</figcaption></figure>
      </div>
    </div>
  </div>

  <!-- History -->
  <div class="card full">
    <h2>📋 Lịch sử phân tích (10 lần gần nhất — click để xem lại)</h2>
    <div id="hist-list"><p style="color:#90A4AE;font-size:.88em">Chưa có ảnh nào được phân tích.</p></div>
  </div>
</div>

<footer>🍎 Fruit Quality Classification · Môn XLATH & TGMT (121036) · ĐH GTVT TP.HCM</footer>

<script>
let selFile=null, history=[];
const emoji={Apple:'🍎',Banana:'🍌',Guava:'🍈',Lime:'🍋',Orange:'🍊'};
const qcls ={Good:'badge-good',Bad:'badge-bad',Mixed:'badge-mixed'};

const drop=document.getElementById('drop');
drop.addEventListener('dragover',e=>{e.preventDefault();drop.classList.add('over')});
drop.addEventListener('dragleave',()=>drop.classList.remove('over'));
drop.addEventListener('drop',e=>{e.preventDefault();drop.classList.remove('over');if(e.dataTransfer.files[0])setFile(e.dataTransfer.files[0])});
document.getElementById('file-in').addEventListener('change',e=>{if(e.target.files[0])setFile(e.target.files[0])});

function setFile(f){
  selFile=f;
  const r=new FileReader();
  r.onload=ev=>{
    document.getElementById('prev-img').src=ev.target.result;
    document.getElementById('prev-wrap').style.display='block';
    document.getElementById('file-name').textContent=`${f.name} (${(f.size/1024).toFixed(1)} KB)`;
    document.getElementById('btn-analyze').disabled=false;
    document.getElementById('result-section').style.display='none';
    document.getElementById('empty').style.display='block';
  };
  r.readAsDataURL(f);
}

function reset_(){
  selFile=null;
  document.getElementById('prev-wrap').style.display='none';
  document.getElementById('btn-analyze').disabled=true;
  document.getElementById('result-section').style.display='none';
  document.getElementById('empty').style.display='block';
  document.getElementById('file-in').value='';
  document.getElementById('prog-bar').style.width='0';
  document.getElementById('prog-wrap').style.display='none';
}

async function analyze(){
  if(!selFile)return;
  document.getElementById('spinner').style.display='block';
  document.getElementById('btn-analyze').disabled=true;
  document.getElementById('empty').style.display='none';
  document.getElementById('result-section').style.display='none';
  document.getElementById('prog-wrap').style.display='block';

  // Animate progress
  let prog=0;
  const pt=setInterval(()=>{
    prog=Math.min(prog+Math.random()*8,88);
    document.getElementById('prog-bar').style.width=prog+'%';
  },200);

  const fd=new FormData(); fd.append('image',selFile);
  try{
    const res=await fetch('/predict',{method:'POST',body:fd});
    const d=await res.json();
    clearInterval(pt);
    document.getElementById('prog-bar').style.width='100%';
    if(!res.ok||d.error){alert('❌ '+d.error);return}
    renderResult(d); addHist(d,selFile.name);
  }catch(e){alert('Lỗi kết nối: '+e.message)}
  finally{
    document.getElementById('spinner').style.display='none';
    document.getElementById('btn-analyze').disabled=false;
    clearInterval(pt);
  }
}

function renderResult(d){
  document.getElementById('empty').style.display='none';
  document.getElementById('result-section').style.display='block';
  const em=emoji[d.fruit]||'🍒';
  const bc=qcls[d.quality]||'badge-mixed';
  const badgeEl=document.getElementById('badge');
  badgeEl.className='badge '+bc;
  badgeEl.innerHTML=`${em} ${d.prediction}`;
  document.getElementById('r-fruit').textContent=d.fruit;
  document.getElementById('r-quality').textContent=d.quality;
  document.getElementById('r-conf').textContent=(d.confidence*100).toFixed(1)+'%';
  document.getElementById('r-defect').textContent=(d.defect_ratio*100).toFixed(2)+'%';
  document.getElementById('r-time').textContent=d.processing_ms+' ms';
  const ul=document.getElementById('top3'); ul.innerHTML='';
  d.top3.forEach((item,i)=>{
    const pct=(item[1]*100).toFixed(1);
    const ic=i===0?'🥇':i===1?'🥈':'🥉';
    ul.innerHTML+=`<li><span>${ic} ${item[0]}</span>
      <span style="display:flex;align-items:center;gap:7px">
        <span class="pbar" style="width:${Math.round(item[1]*80)}px"></span><b>${pct}%</b>
      </span></li>`;
  });
  document.getElementById('img-orig').src='data:image/png;base64,'+d.img_original;
  document.getElementById('img-res').src ='data:image/png;base64,'+d.img_result;
}

function addHist(d,fname){
  history.unshift({d,fname,ts:new Date().toLocaleTimeString()});
  if(history.length>10)history.pop();
  const c=document.getElementById('hist-list'); c.innerHTML='';
  history.forEach(h=>{
    const em=emoji[h.d.fruit]||'🍒';
    const el=document.createElement('div');
    el.className='hi'; el.onclick=()=>renderResult(h.d);
    el.innerHTML=`<img src="data:image/png;base64,${h.d.img_original}" alt="">
      <div class="inf"><strong>${em} ${h.d.prediction}</strong>
      <small>${h.fname} · ${h.ts} · Conf:${(h.d.confidence*100).toFixed(0)}%</small></div>`;
    c.appendChild(el);
  });
}
</script>
</body>
</html>"""

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/health')
def health():
    return jsonify({'status':'ok','model':os.path.exists(MODEL_PATH),'classes':ALL_CLASSES})

@app.route('/predict', methods=['POST'])
def predict():
    t0 = time.time()
    if 'image' not in request.files:
        return jsonify({'error':'Không tìm thấy file ảnh'}), 400
    file = request.files['image']
    if not file.filename:
        return jsonify({'error':'Tên file rỗng'}), 400

    raw_bytes = np.frombuffer(file.read(), np.uint8)
    img_bgr   = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return jsonify({'error':'Không đọc được ảnh — dùng JPG/PNG/BMP'}), 400

    try:
        data     = get_model()
        pipeline = data['pipeline']

        img  = load_resize(img_bgr)
        enh, gray = preprocess(img)

        # Segmentation
        try:
            fm  = segment_grabcut(enh)
            dm, dr = detect_defects(enh, fm)
        except Exception:
            fm = np.ones(enh.shape[:2],np.uint8)*255
            dm = np.zeros(enh.shape[:2],np.uint8)
            dr = 0.0

        feat     = extract_features(enh, fm, dr)
        pred_idx = int(pipeline.predict(feat.reshape(1,-1))[0])
        proba    = pipeline.predict_proba(feat.reshape(1,-1))[0]
        pred     = IDX_TO_CLASS.get(pred_idx,'Unknown')
        parts    = pred.split('_') if '_' in pred else [pred,'?']
        fruit, quality = parts[0], (parts[1] if len(parts)>1 else '?')
        conf     = float(proba[pred_idx]) if pred_idx<len(proba) else 0.0
        top3     = [(IDX_TO_CLASS.get(int(i),'?'),float(proba[i]))
                    for i in np.argsort(proba)[::-1][:3]]

        vis = render_vis(enh, fm, dm, pred, conf, dr)

        return jsonify({
            'prediction':    pred,
            'fruit':         fruit,
            'quality':       quality,
            'confidence':    round(conf,4),
            'defect_ratio':  round(dr,4),
            'top3':          top3,
            'processing_ms': round((time.time()-t0)*1000,1),
            'img_original':  img_to_b64(img),
            'img_result':    img_to_b64(vis),
        })

    except FileNotFoundError as e:
        return jsonify({'error':str(e)}), 503
    except Exception as e:
        return jsonify({'error':f'Lỗi: {e}','detail':traceback.format_exc()}), 500

if __name__ == '__main__':
    print('='*55)
    print('  🍎 Fruit Quality Inspection — Web Server')
    print('='*55)
    if os.path.exists(MODEL_PATH):
        print(f'  ✅ Model: {MODEL_PATH}')
    else:
        print(f'  ⚠️  Chưa có model — chạy notebook trước!')
    print('  🌐 http://localhost:5000')
    print('='*55)
    app.run(debug=True, host='0.0.0.0', port=5000)
