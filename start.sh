#!/bin/bash
# 渔船作业检测 API 启动脚本

echo "========================================="
echo "  渔船作业检测 API 服务启动"
echo "========================================="

# 检查 Python
if ! command -v python3 &> /dev/null; then
    echo "❌ 未找到 python3，请先安装 Python 3.8+"
    exit 1
fi

# 安装依赖
echo ""
echo "📦 安装图片检测 API 依赖..."
cd image_api && pip install -r requirements.txt -q && cd ..

echo "📦 安装视频分析 API 依赖..."
cd video_api && pip install -r requirements.txt -q && cd ..

# 创建日志目录
mkdir -p image_api/logs
mkdir -p video_api/logs

echo ""
echo "🚀 启动图片检测 API (端口 8001)..."
cd image_api && python3 api_server.py &
IMAGE_PID=$!
cd ..

echo "🚀 启动视频分析 API (端口 8002)..."
cd video_api && python3 video_api_server.py &
VIDEO_PID=$!
cd ..

echo ""
echo "✅ 服务已启动！"
echo "   图片检测 API: http://localhost:8001/infer"
echo "   视频分析 API: http://localhost:8002/infer_video"
echo "   API Key: 123"
echo ""
echo "   图片检测 PID: $IMAGE_PID"
echo "   视频分析 PID: $VIDEO_PID"
echo ""
echo "按 Ctrl+C 停止所有服务"

# 等待子进程
wait
