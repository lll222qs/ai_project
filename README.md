# 👕 Fashion-MNIST 智能分类器 | 全栈 AI 应用

一个基于现代 Web 技术的时尚单品图像分类系统，展示从数据训练到生产级部署的完整 AI 工程化流程。

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](http://localhost:8501)
![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.119.1-green.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0.0-red.svg)
![SQLAlchemy](https://img.shields.io/badge/SQLAlchemy-2.0.0-yellow.svg)
![GitHub Stars](https://img.shields.io/github/stars/lll222qs/ai_project.svg?style=social)

---

## ✨ 核心特性

- **🎯 精准识别**：基于深度学习模型，准确识别 10 类时尚单品（T恤、裤子、鞋子等）
- **🚀 多模型推理**：支持 PyTorch、TorchScript、量化模型、ONNX 四种推理引擎
- **🔒 生产级架构**：完整的 API 认证、错误处理、日志记录和监控统计
- **📊 数据持久化**：SQLite 数据库记录所有预测历史，支持数据分析和回溯
- **⚡ 性能对比**：内置基准测试系统，实时对比不同模型推理性能
- **🎨 现代化前端**：Streamlit 构建的交互式 Web 界面，支持实时可视化
- **🔧 一键部署**：Python 服务编排系统，实现多服务统一管理和监控

---

## 🏗️ 系统架构
用户界面 (Streamlit)
↓ HTTP 请求
服务网关 (FastAPI + 认证中间件 + 日志中间件)
↓ 模型路由
多模型推理引擎
├── PyTorch (原始模型)
├── TorchScript (静态图优化)
├── 动态量化模型 (体积优化)
└── ONNX Runtime (跨框架部署)
↓ 数据持久化
存储层 (SQLite + Redis)
├── 预测记录持久化
└── 实时统计缓存



---

## 🛠️ 技术栈

| 领域 | 技术选型 | 核心用途 |
|------|----------|----------|
| **前端界面** | Streamlit, Plotly, Pillow | Web 交互、数据可视化、图片处理 |
| **后端服务** | FastAPI, Uvicorn, Pydantic | RESTful API、数据验证、异步处理 |
| **AI 框架** | PyTorch, TorchScript, ONNX Runtime | 模型推理、性能优化、跨平台部署 |
| **数据存储** | SQLite, SQLAlchemy, Redis | 数据持久化、ORM 映射、缓存统计 |
| **服务编排** | Python subprocess, 信号处理 | 多服务管理、进程监控、优雅停机 |
| **工程工具** | Git, logging, argparse | 版本控制、结构化日志、参数解析 |

---

## 🚀 快速开始

### 1. 本地运行（推荐）

#### 环境要求
- Python 3.9+
- 推荐使用虚拟环境

#### 安装依赖
```bash
git clone https://github.com/lll222qs/ai_project.git
cd ai_project
pip install -r requirements.txt

一键启动所有服务：
# 使用服务编排器启动完整系统
python scripts/orchestrator.py

手动启动（开发模式）：
# 终端1：启动后端API
uvicorn app.main:app --reload --port 8000

# 终端2：启动前端界面  
streamlit run streamlit_app_v5.py --server.port 8501

2. 访问服务
服务启动后，访问以下地址：

前端界面: http://localhost:8501

API 文档: http://localhost:8000/docs

系统统计: http://localhost:8000/stats


📁 项目结构:
my_fastapi_project/
├── app/                          # 后端核心模块
│   ├── __init__.py
│   ├── main.py                   # FastAPI 主应用（V5完全体）
│   ├── database.py               # 数据库配置和会话管理
│   ├── models.py                 # SQLAlchemy 数据模型
│   └── app.log                   # 应用日志
├── scripts/                      # 部署和管理脚本
│   ├── orchestrator.py           # 服务编排管理器（核心！）
│   ├── start_services.sh         # 服务启动脚本
│   └── stop_services.sh          # 服务停止脚本
├── data/                         # 数据文件目录
│   └── predictions.db            # SQLite 数据库文件
├── streamlit_app_v5.py           # Streamlit 前端（V5完全体）
├── requirements.txt              # Python 依赖列表
├── README.md                     # 项目说明文档
└── .gitignore                    # Git 忽略规则


🔌 API 接口
认证方式
所有预测接口都需要 API Key 认证：
X-API-Key: 123456

核心接口
1. 多模型预测接口
POST /predict - 原始 PyTorch 模型

POST /predict-scripted - TorchScript 优化模型

POST /predict-quantized - 动态量化模型

POST /predict-onnx - ONNX Runtime 模型

请求示例：
curl -X POST "http://localhost:8000/predict" \
  -H "X-API-Key: 123456" \
  -F "file=@test_image.jpg"
  响应示例：
{
  "model": "PyTorch",
  "class": "T-shirt/top", 
  "confidence": 0.9562,
  "inference_time_ms": 12.34
}
2. 性能基准测试
POST /predict-benchmark
对比所有可用模型的推理性能，返回详细的速度分析报告。
3. 系统监控
GET /stats
获取系统统计信息（总预测次数、服务状态等）。


🎯 功能演示
前端界面功能
多模型预测 - 选择不同优化模型进行图像分类

性能对比 - 基准测试对比各模型推理速度

实时可视化 - 置信度分布图表和性能图表

系统监控 - 服务状态和统计信息展示

后端特性
自动服务编排 - 一键启动数据库、API、前端服务

进程监控 - 实时监控服务状态，异常自动处理

优雅停机 - Ctrl+C 安全停止所有服务

结构化日志 - 完整的请求链路追踪


🔧 部署说明
服务编排系统
项目内置了完整的服务编排解决方案，无需 Docker 即可实现生产级部署：
# 服务编排器核心功能
- 数据库自动初始化
- 多进程服务管理
- 健康状态监控
- 优雅停机处理
- 错误恢复机制


启动方式
# 开发环境
python scripts/orchestrator.py

# 生产环境  
./scripts/start_services.sh


📊 技术亮点
1. 工程化实践
✅ 服务编排与进程管理

✅ 数据库集成与 ORM 映射

✅ API 认证与安全防护

✅ 结构化日志与错误处理

✅ 后台任务与异步处理

2. AI 工程化
✅ 多模型推理框架

✅ 性能基准测试系统

✅ 模型优化技术对比

✅ 跨框架部署方案

3. 系统设计
✅ 前后端分离架构

✅ 微服务化设计

✅ 配置化管理

✅ 监控统计体系


📄 许可证
本项目采用 Apache 2.0 许可证 - 查看 LICENSE 文件了解详情。


📞 联系方式
开发者: lll222qs

邮箱: 1757952556@qq.com

项目地址: https://github.com/lll222qs/ai_project
如果这个项目对你有帮助，请给个 ⭐️ 星标支持！


