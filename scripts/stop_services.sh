#!/bin/bash

echo "🛑 停止所有服务..."

# 停止所有Python相关服务
pkill -f "uvicorn main:app"
pkill -f "streamlit run streamlit_app.py" 
pkill -f "python -m http.server 8001"

echo "✅ 所有服务已停止"