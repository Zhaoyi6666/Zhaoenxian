# 渔船作业检测 API

本项目包含两个 API 服务，用于渔船作业状态检测。

## 项目结构

```
.
├── image_api/          # 图片目标检测 API (YOLO)
│   ├── api_server.py
│   ├── best_fishing_model.pt
│   ├── requirements.txt
│   └── 接口文档.md
├── video_api/          # 视频作业姿态分析 API (Qwen2.5-VL)
│   ├── video_api_server.py
│   ├── requirements.txt
│   └── 接口文档.md
├── start.sh            # 一键启动脚本
└── README.md
```

## 环境要求

- Python 3.8+
- NVIDIA GPU（建议显存 >= 24G，用于视频分析模型）
- 图片检测 API 可在 CPU 上运行（速度较慢）

## 快速启动

```bash
# 安装依赖并启动所有服务
chmod +x start.sh
./start.sh
```

或分别启动：

```bash
# 图片检测 API (端口 8001)
cd image_api
pip install -r requirements.txt
python api_server.py

# 视频分析 API (端口 8002)
cd ../video_api
pip install -r requirements.txt
python video_api_server.py
```

## 接口说明

### 图片检测 API

- **地址**: `POST http://localhost:8001/infer`
- **Header**: `Authorization: Bearer 123`
- **Body**: `multipart/form-data`，字段 `file`（图片，JPG/PNG，≤10MB）

### 视频分析 API

- **地址**: `POST http://localhost:8002/infer_video`
- **Header**: `Authorization: Bearer 123`
- **Body**: `multipart/form-data`，字段 `file`（视频，建议 10-30 秒）

> ⚠️ 视频分析 API 需要下载 Qwen2.5-VL 模型，默认路径 `/mnt/workspace/qwen2.5-vl-7b`，请根据实际情况修改 `video_api_server.py` 中的 `MODEL_PATH`。

## 日志

- 图片检测日志: `image_api/logs/api_call.jsonl`
- 视频分析日志: `video_api/logs/video_api_call.jsonl`
