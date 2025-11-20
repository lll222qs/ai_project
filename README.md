👕 Fashion-MNIST 智能分类器 | 全栈 AI 应用

image

image

image

image

image
一个基于现代 Web 技术的时尚单品图像分类系统，展示从数据训练到生产级部署的完整 AI 工程化流程，新增模型优化与性能对比功能，进一步提升系统工业级应用能力。
✨ 项目特色
🎯 精准识别 - 基于深度学习模型，准确识别 10 类时尚单品（T 恤、裤子、鞋子等）
🚀 实时预测 - 毫秒级响应，上传图片即刻获得分类结果及置信度
🔍 模型优化 - 新增 TorchScript 模型优化方案，提供更快的推理速度
🔒 安全访问 - 完善的 API 密钥认证机制，防止接口被恶意滥用
📊 性能对比 - 支持原始 PyTorch 模型与 TorchScript 优化模型的推理速度对比
📝 可观测性 - 集成结构化日志系统，记录请求 IP、时间、处理耗时等关键信息
🎨 友好界面 - 简洁直观的 Web 操作界面，支持拖拽上传图片
🔧 标准 API - 符合 REST 规范的接口设计，易于集成到其他系统
📱 跨端兼容 - 完美支持桌面和移动设备访问
🏗 系统架构
text
📱 用户界面 (Streamlit) 
    ↓ （HTTP请求）
🌐 REST API (FastAPI + Uvicorn)
    ├─ 🔒 认证中间件（API密钥验证）
    ├─ 📝 日志中间件（请求记录）
    ↓
🧠 AI引擎 (PyTorch + TorchScript优化)
    ├─ 原始PyTorch模型推理
    ├─ TorchScript优化模型推理
    └─ 模型性能对比模块
    ↓
🖼 图像处理 (Pillow + OpenCV)
🛠 技术栈
领域	技术选型	新增特性
前端界面	Streamlit, Pillow	-
后端服务	FastAPI, Uvicorn, Pydantic	API 密钥认证、请求日志中间件
AI 框架	PyTorch, Torchvision, NumPy	TorchScript 模型优化、推理性能对比
工程化	Docker, Git, pip	结构化日志（logging 模块）、CPU 并行计算优化
部署	Streamlit Cloud, Hugging Face Spaces	-
🚀 快速开始
在线体验
无需安装，立即在浏览器中体验 AI 图像分类：
演示地址：https://fashion-mnist-classifier-my8yyomkawax5v6xjupzj3.streamlit.app/
本地运行
bash
运行
# 1. 克隆项目
git clone https://github.com/lll222qs/ai_project.git
cd ai_project

# 2. 安装依赖
pip install -r requirements.txt

# 3. 启动后端服务（带自动重载）
uvicorn main_v3:app --reload

# 4. 启动前端界面（新终端）
streamlit run streamlit_app.py
📁 项目结构
text
ai_project/
├── main_v3.py              # FastAPI后端主程序（含TorchScript优化、性能对比）
├── main_v2.py              # 历史版本：基础版后端实现
├── streamlit_app.py        # Streamlit前端界面
├── requirements.txt        # 项目依赖清单
├── Dockerfile              # 容器化配置
├── fashion_model.pth       # 训练好的PyTorch模型
├── fashion_model_scripted.pt # TorchScript优化模型
├── app.log                 # 自动生成的结构化日志文件（运行后出现）
└── README.md               # 项目说明文档
🎯 核心功能
图像分类
支持 PNG、JPG、JPEG 格式图片上传
自动完成图像预处理（灰度转换、尺寸调整、对比度优化）
返回分类结果及置信度（保留 4 位小数）
模型优化
提供原始 PyTorch 模型与 TorchScript 优化模型双接口
支持模型推理速度对比，直观展示优化效果
内置 CPU 并行计算优化配置，可根据硬件调整线程数
安全认证
所有 API 请求需携带 X-API-Key 请求头
支持多密钥配置，便于权限管理
非法请求自动拦截并返回 401 错误
日志监控
记录客户端 IP、请求方法、路径、处理耗时、状态码
自动记录预测成功 / 失败详情（含文件名、错误信息）
日志文件自动轮转（最大 10MB，保留 3 个备份）
🔌 API 接口文档
启动后端服务后，访问http://127.0.0.1:8000/docs可查看交互式 API 文档，核心接口说明：
预测接口（需认证）
POST /predict（原始 PyTorch 模型）
POST /predict-scripted（TorchScript 优化模型）
POST /predict-benchmark（模型性能对比）
请求参数：
Content-Type: multipart/form-data
Headers: X-API-Key: your_secure_key_123
Body: 图片文件
响应示例：
json
{
    "model": "TorchScript",
    "class": "T-shirt/top",
    "confidence": 0.9876,
    "inference_time_ms": 2.34
}
🔮 未来规划
支持批量图片预测与压缩包上传
增加模型性能监控面板（响应时间、准确率统计）
实现用户历史记录与结果导出功能
集成模型再训练流水线，支持自定义数据集
🤝 贡献指南
我们欢迎任何形式的贡献！请参考以下步骤：
Fork 本仓库
创建特性分支 (git checkout -b feature/AmazingFeature)
提交更改 (git commit -m 'Add some AmazingFeature')
推送到分支 (git push origin feature/AmazingFeature)
开启 Pull Request
📄 许可证
本项目采用 Apache 2.0 许可证 - 查看 LICENSE 文件了解详情
📞 联系我们
📧 邮箱：1757952556@qq.com
💻 GitHub: lll222qs
🎯 项目地址：https://github.com/lll222qs/ai_project