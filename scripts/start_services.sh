#!/bin/bash

echo "🚀 启动AI应用集群..."

# 启动数据库服务
echo "启动数据库..."
python -m http.server 8001 &
DB_PID=$!

# 等待数据库就绪
sleep 2

# 启动FastAPI服务
echo "启动FastAPI后端..."
uvicorn main:app --host 0.0.0.0 --port 8000 &
API_PID=$!

# 启动Streamlit前端
echo "启动Streamlit前端..."
streamlit run streamlit_app_v5.py --server.port 8501 --server.address 0.0.0.0 &
WEB_PID=$!

echo "✅ 所有服务启动完成!"
echo "📊 访问地址:"
echo "   - API文档: http://localhost:8000/docs"
echo "   - Web界面: http://localhost:8501"

# 等待用户中断
echo "按 Ctrl+C 停止所有服务"
wait