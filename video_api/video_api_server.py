import os
import io
import re
import cv2
import json
import time
import tempfile
import torch
from fastapi import FastAPI, File, UploadFile, Header, HTTPException
from fastapi.responses import JSONResponse
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import uvicorn

MODEL_PATH = "/mnt/workspace/qwen2.5-vl-7b" # 模型路径
API_KEY = "123"
LOG_DIR = "./logs"
os.makedirs(LOG_DIR, exist_ok=True)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

print("⏳ 正在加载 Qwen2.5-VL 模型...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
)
processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
print("✅ 模型加载完成！")

app = FastAPI()

PROMPT_TEXT = """你是一位渔业执法专家。请仔细观察这组连续画面，分析该渔船的作业状态。
【注意】画面可能经过加速处理，动作较快。请重点观察甲板机械运作、网具形态以及船只的航行动作。

【业务规则】
1. 只要画面中【出现船员操作网具、绳索，或有放网、收网、拖曳等捕捞动作迹象】，即使动作很快，也必须判定为对应的作业姿态（放网、捕捞、收网），并尽量识别作业类型（拖网、围网、刺网等）。
2. 只有在【完全没有任何捕捞动作迹象、没有任何网具出现、船只只是平稳航行】的情况下，才判定为“正常航行姿态”，且作业类型统一写“无”。禁止输出“存疑”。

【第一步：通用作业阶段姿态（选一）】
1. 正常航行姿态 2. 放网姿态 3. 捕捞作业姿态 4. 收网姿态 5. 停泊漂流姿态

【第二步：作业类型（选一）】
拖网、围网、刺网、钓具、笼壶/陷阱、耙刺。如果判定为正常航行，作业类型统一写“无”。

【输出格式】
- 通用姿态：
- 作业类型：
- 视觉特征分析：（20字以内）
- 最终结论：（一句话综合描述）"""

def verify_token(authorization: str = Header(None)):
    if not authorization or authorization.split(" ")[-1] != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")

def write_log(filename: str, status: str, result: dict, elapsed_ms: float):
    log_entry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "filename": filename,
        "status": status,
        "result": result,
        "elapsed_ms": round(elapsed_ms, 2)
    }
    with open(os.path.join(LOG_DIR, "video_api_call.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

@app.post("/infer_video")
async def infer_video(file: UploadFile = File(...), authorization: str = Header(None)):
    verify_token(authorization)
    start_time = time.time()

    temp_video_path = os.path.join(tempfile.gettempdir(), file.filename)
    with open(temp_video_path, "wb") as buffer:
        buffer.write(await file.read())

    try:
        cap = cv2.VideoCapture(temp_video_path)
        fps_video = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps_video if fps_video > 0 else 15
        cap.release()

        target_frames = 16
        dynamic_fps = max(0.1, min(2.0, target_frames / duration))

        messages = [
            {"role": "user", "content": [
                {"type": "video", "video": temp_video_path, "min_pixels": 100*28*28, "max_pixels": 360*28*28, "fps": dynamic_fps},
                {"type": "text", "text": PROMPT_TEXT},
            ]}
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=256)

        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        posture_match = re.search(r'通用姿态：(.*)', output_text)
        gear_match = re.search(r'作业类型：(.*)', output_text)

        posture = posture_match.group(1).strip() if posture_match else "正常航行姿态"
        gear_type = gear_match.group(1).strip() if gear_match else "无"

        result = {
            "answer": output_text.strip(),
            "extras": {
                "posture": posture,
                "gear_type": gear_type,
                "duration_analyzed": round(duration, 2),
                "frames_analyzed": target_frames
            }
        }

        elapsed_ms = (time.time() - start_time) * 1000
        write_log(file.filename, "success", result, elapsed_ms)

        return JSONResponse(status_code=200, content=result)

    except Exception as e:
        elapsed_ms = (time.time() - start_time) * 1000
        write_log(file.filename, "error", {"error": str(e)}, elapsed_ms)
        return JSONResponse(status_code=500, content={"answer": f"视频分析失败: {e}", "extras": {"error": str(e)}})
    finally:
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)
