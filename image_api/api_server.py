import io
import os
import cv2
import json
import time
import numpy as np
from datetime import datetime
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Header, HTTPException
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import uvicorn

MODEL_PATH = "./best_fishing_model.pt"
API_KEY = "123"
LOG_DIR = "./logs"

os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "api_call.jsonl")

LABEL_MAP = {0:"渔船", 1:"耙刺", 2:"刺网", 3:"钓具", 4:"笼壶", 5:"拖网", 6:"围网", 7:"陷阱", 8:"杂渔具", 9:"张网", 10:"人"}

model = YOLO(MODEL_PATH)
app = FastAPI()

def write_log(filename: str, status: str, count: int, elapsed_ms: float, error: str = None):
    """写入本地 JSON 行日志"""
    try:
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "filename": filename,
            "status": status,
            "detection_count": count,
            "elapsed_ms": round(elapsed_ms, 2),
            "error": error
        }
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"写入日志失败: {e}")

def verify_token(authorization: str = Header(None)):
    if not authorization or authorization.split(" ")[-1] != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")

@app.post("/infer")
async def infer(file: UploadFile = File(...), authorization: str = Header(None)):
    verify_token(authorization)
    start_time = time.time()

    try:
        contents = await file.read()
        img_pil = Image.open(io.BytesIO(contents)).convert("RGB")
        img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        h, w = img_cv.shape[:2]
    except Exception as e:
        elapsed_ms = (time.time() - start_time) * 1000
        write_log(file.filename, "decode_error", 0, elapsed_ms, str(e))
        return JSONResponse(status_code=400, content={"answer": f"图片解析失败: {e}", "extras": {"empty": True, "count": 0}})

    results = model(img_cv, conf=0.25, verbose=False)[0]
    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        detections.append({
            "label": LABEL_MAP.get(int(box.cls[0]), "未知"),
            "confidence": round(float(box.conf[0]), 4),
            "bbox_xyxy": [x1, y1, x2, y2]
        })

    count = len(detections)
    answer = "未在画面中检出目标网具或作业目标。" if count == 0 else f"检出 {count} 个目标。"

    elapsed_ms = (time.time() - start_time) * 1000
    write_log(file.filename, "success", count, elapsed_ms) # 记录成功日志

    return JSONResponse(status_code=200, content={
        "answer": answer,
        "extras": {
            "empty": count == 0, "count": count,
            "image_width": w, "image_height": h,
            "detections": detections
        }
    })

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
