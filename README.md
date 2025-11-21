# 👕 Fashion-MNIST 智能分类器 | 全栈 AI 应用

一个基于现代 Web 技术的时尚单品图像分类系统，展示从数据训练到生产级部署的完整 AI 工程化流程。

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://fashion-mnist-classifier-my8yyomkawax5v6xjupzj3.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.68.0-green.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0.0-red.svg)
![ONNX](https://img.shields.io/badge/ONNX-1.14.0-orange.svg)
![Docker](https://img.shields.io/badge/Docker-20.10+-blue.svg)
![GitHub Stars](https://img.shields.io/github/stars/lll222qs/ai_project.svg?style=social)

---

## 📸 项目演示

![项目主界面](https://via.placeholder.com/800x450?text=Fashion-MNIST+Classifier+Main+UI)
*上图为项目的 Streamlit Web 界面，支持图片拖拽上传和实时分类结果展示。*

![API 文档界面](https://via.placeholder.com/800x450?text=FastAPI+Swagger+UI)
*上图为 FastAPI 自动生成的交互式 API 文档，方便开发者调试。*

---

## ✨ 核心特性

- **🎯 精准识别**：基于深度学习模型，准确识别 10 类时尚单品（T恤、裤子、鞋子等）。
- **🚀 极致性能**：
  - **TorchScript**：静态图优化，推理速度提升 2-3 倍。
  - **ONNX**：跨框架部署，适配 CPU/GPU，速度再提升 5-10 倍。
  - **动态量化**：模型体积缩减 75%，CPU 推理更高效。
- **🔒 安全防护**：API 密钥认证，支持多密钥权限分级，非法请求自动拦截。
- **📊 可观测性**：
  - **结构化日志**：记录请求 IP、耗时、结果等全链路信息。
  - **Redis 统计**：实时监控总预测次数。
  - **数据库持久化**：所有预测记录存入 SQLite，支持后续分析。
- **⚡ 性能对比**：内置基准测试接口，一键对比 4 种模型推理速度。
- **🎨 友好界面**：Streamlit 可视化交互，支持拖拽上传和结果展示。
- **🔧 工程化部署**：Docker 容器化封装，支持本地/云服务器快速部署。

---

## 🏗️ 系统架构
┌───────────────┐ HTTP ┌──────────────────────────────────────┐│ 
│◄────────────►│ FastAPI │
│ Streamlit │ Requests │ ┌──────────┐ ┌──────────┐ ┌──────┐ │
│ (前端) │ │ │ 认证中间件 │ │ 日志中间件 │ │Redis │ │
│ │ │ └──────────┘ └──────────┘ └──────┘ │
└───────────────┘ └──────────────────┬───────────────────┘│┌───────────────────────┼───────────────────────┐
│ 
│ │
┌───────▼───────┐ ┌───────▼───────┐ ┌──────▼───────┐
│ PyTorch │ │ TorchScript │ │ ONNX │
│ (原始模型) │ │ (静态图优化) │ │ (跨框架) │
└───────┬───────┘ └───────┬───────┘ └──────┬───────┘
│ │ │
└───────────────────────┼───────────────────────┘
│
┌───────▼───────┐
│ 图像处理 │
│ (Pillow/OpenCV)│
└───────────────┘

---

## 🛠️ 技术栈

| 领域         | 技术选型                          | 核心特性                                  |
|--------------|-----------------------------------|-------------------------------------------|
| **前端**     | Streamlit, Pillow                 | 拖拽上传、实时可视化、跨端兼容            |
| **后端**     | FastAPI, Uvicorn, Pydantic        | API 认证、请求日志、异步处理、自动生成文档 |
| **AI 框架**  | PyTorch, TorchVision, ONNX Runtime | 多模型支持、推理优化、跨框架部署          |
| **数据存储** | SQLite, Redis, SQLAlchemy         | 持久化存储、缓存统计、ORM 映射            |
| **工程化**   | Docker, Git, pip                  | 容器化部署、版本控制、依赖管理            |
| **部署**     | Streamlit Cloud, Hugging Face     | 在线演示、快速部署、公开访问              |

---

## 🚀 快速开始

### 1. 在线体验

无需安装，立即在浏览器中体验：  
[Fashion-MNIST Classifier Demo](https://fashion-mnist-classifier-my8yyomkawax5v6xjupzj3.streamlit.app/)

### 2. 本地运行

#### 环境准备

- Python 3.9+
- Git
- Redis (可选，用于统计功能)

#### 克隆项目

```bash
 git clone https://github.com/lll222qs/ai_project.git
cd ai_project
安装依赖:
pip install -r requirements.txt
准备模型文件
确保项目根目录下有以下模型文件：
fashion_model.pth (原始 PyTorch 模型)
fashion_model_scripted.pt (TorchScript 模型)
fashion_model.onnx (ONNX 模型)
如果没有，可以运行 export_torchscript.py 和 export_onnx.py 脚本生成。
启动服务
# 启动后端 API 服务 (默认端口 8000)
uvicorn app.main_v5:app --reload

# 在新终端启动前端 Web 界面 (默认端口 8501)
streamlit run streamlit_app.py

访问服务
前端界面: http://localhost:8501
API 文档: http://localhost:8000/docs (交互式 Swagger UI)
统计接口: http://localhost:8000/stats

📁 项目结构
ai_project/
├── app/                          # 后端核心模块
│   ├── __init__.py               # Python 包标识
│   ├── main_v5.py                # 第五版主程序 (多模型+DB+Redis)
│   ├── database.py               # 数据库连接管理 (SQLAlchemy)
│   ├── models.py                 # 数据模型 (Prediction 表结构)
│   └── app.log                   # 结构化日志文件
├── streamlit_app.py              # 前端交互界面
├── export_onnx.py                # ONNX 模型导出脚本
├── export_torchscript.py         # TorchScript 模型导出脚本
├── requirements.txt              # 项目依赖清单
├── Dockerfile                    # 容器化配置文件
├── .gitignore                    # Git 忽略规则
├── fashion_model.pth             # 原始 PyTorch 模型
├── fashion_model_scripted.pt     # TorchScript 优化模型
├── fashion_model.onnx            # ONNX 跨框架模型
├── test.db                       # SQLite 数据库文件
└── README.md                     # 项目说明文档

🔌 API 接口文档
启动后端服务后，访问 http://localhost:8000/docs 可查看交互式 API 文档。
核心接口
1. 预测接口 (需认证)
POST /predict: 使用原始 PyTorch 模型预测
POST /predict-scripted: 使用 TorchScript 模型预测
POST /predict-onnx: 使用 ONNX 模型预测
POST /predict-quantized: 使用动态量化模型预测

请求示例:
curl -X POST "http://localhost:8000/predict-onnx" \
     -H "X-API-Key: 123456" \
     -F "file=@/path/to/your/image.jpg"

2. 性能基准测试接口
POST /predict-benchmark: 对比所有已加载模型的推理速度

响应示例:
{
  "filename": "test_shirt.jpg",
  "models_tested": ["PyTorch", "TorchScript", "ONNX", "PyTorch (Quantized)"],
  "inference_times_ms": {
    "PyTorch": 15.625,
    "TorchScript": 6.25,
    "ONNX": 3.125,
    "PyTorch (Quantized)": 7.8125
  },
  "speed_comparison": {
    "PyTorch": "1.00x",
    "TorchScript": "2.50x",
    "ONNX": "5.00x",
    "PyTorch (Quantized)": "2.00x"
  },
  "fastest_model": {
    "name": "ONNX",
    "time_ms": 3.125,
    "speedup_over_slowest": "5.00x"
  }
}

🔮 未来规划
 支持批量图片预测与压缩包上传 (ZIP/RAR)
 增加模型性能监控面板 (响应时间、准确率统计可视化)
 实现用户历史预测记录查询与结果导出 (CSV/Excel)
 集成模型再训练流水线，支持自定义数据集更新模型
 部署到 Kubernetes 集群，支持自动扩缩容与高可用

 🤝 贡献指南
Fork 本仓库
创建特性分支 (git checkout -b feature/amazing-feature)
提交更改 (git commit -m 'Add some amazing feature')
推送到分支 (git push origin feature/amazing-feature)
开启 Pull Request
📄 许可证
本项目采用 Apache 2.0 许可证 - 开源可商用，查看 LICENSE 文件了解详情。

📞 联系我们
📧 邮箱: 1757952556@qq.com
💻 GitHub: lll222qs
🎯 项目地址: https://github.com/lll222qs/ai_project
如果这个项目对你有帮助，请给个 ⭐️ 星标支持！