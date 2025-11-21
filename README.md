👕 Fashion-MNIST 智能分类器 | 全栈 AI 应用

image

image

image

image

image

image
一个基于现代 Web 技术的时尚单品图像分类系统，展示从数据训练到生产级部署的完整 AI 工程化流程。新增多模型优化（PyTorch/TorchScript/ONNX/ 动态量化）、数据库持久化、Redis 缓存统计及性能基准测试功能，完全贴合工业级应用标准。
✨ 项目特色
🎯 精准识别：基于深度学习模型，准确识别 10 类时尚单品（T 恤、裤子、鞋子等），支持多模型推理对比
🚀 极致性能：
TorchScript 静态图优化，推理速度提升 2-3 倍
ONNX 跨框架部署，适配 CPU/GPU 硬件加速，速度再提升 5-10 倍
8 位动态量化，模型体积缩减 75%，CPU 推理更高效
🔒 安全防护：API 密钥认证机制，支持多密钥权限分级，非法请求自动拦截
📊 可观测性：
结构化日志系统，记录请求 IP、耗时、结果等全链路信息
Redis 实时统计总预测次数，支持可视化查询
数据库持久化存储所有预测记录，支持后续分析
⚡ 性能对比：内置基准测试接口，一键对比 4 种模型推理速度，生成直观加速比报告
🎨 友好界面：Streamlit 可视化交互，支持拖拽上传、实时展示分类结果及置信度
🔧 工程化部署：Docker 容器化封装，支持本地 / 云服务器快速部署
🏗 系统架构
text
📱 用户界面 (Streamlit) 
    ↓ （HTTP请求）
🌐 REST API (FastAPI + Uvicorn)
    ├─ 🔒 认证中间件（API密钥验证）
    ├─ 📝 日志中间件（全链路请求记录）
    ├─ 📊 缓存统计（Redis总预测次数）
    └─ 📦 数据库交互（SQLite持久化存储）
    ↓
🧠 AI引擎（多模型推理层）
    ├─ PyTorch 原始模型
    ├─ TorchScript 静态图优化模型
    ├─ ONNX Runtime 跨框架模型
    └─ PyTorch 动态量化模型
    ↓
🖼 图像处理 (Pillow + TorchVision)
    └─ 灰度转换→尺寸调整→对比度优化→张量归一化
🛠 技术栈
领域	技术选型	核心特性
前端界面	Streamlit、Pillow	拖拽上传、实时可视化、跨端兼容
后端服务	FastAPI、Uvicorn、Pydantic	API 认证、请求日志、异步处理、自动生成文档
AI 框架	PyTorch、TorchVision、ONNX Runtime	多模型支持、推理优化、跨框架部署
数据存储	SQLite、Redis、SQLAlchemy	持久化存储、缓存统计、ORM 映射
工程化工具	Docker、Git、pip	容器化部署、版本控制、依赖管理
部署平台	Streamlit Cloud、Hugging Face	在线演示、快速部署、公开访问
🚀 快速开始
在线体验
无需安装，立即在浏览器中体验 AI 图像分类：演示地址
本地运行
1. 环境准备
Python 3.9+
Git
Redis（本地安装或使用云 Redis 服务）
2. 克隆项目
bash
运行
git clone https://github.com/lll222qs/ai_project.git
cd ai_project
3. 安装依赖
bash
运行
pip install -r requirements.txt
4. 准备模型文件
已训练的 PyTorch 模型：fashion_model.pth（放在项目根目录）
生成 TorchScript 模型：
bash
运行
python export_torchscript.py  # 需提前编写模型导出脚本
生成 ONNX 模型：
bash
运行
python export_onnx.py  # 已集成TorchDynamo优化
5. 启动服务
bash
运行
# 启动后端服务（支持自动重载）
uvicorn app.main_v5:app --reload

# 启动前端界面（新终端执行）
streamlit run streamlit_app.py
6. 访问服务
前端界面：http://localhost:8501
API 文档：http://localhost:8000/docs（支持交互式测试）
统计接口：http://localhost:8000/stats（查看总预测次数）
📁 项目结构
plaintext
ai_project/
├── app/                          # 后端核心模块
│   ├── __init__.py               # Python包标识
│   ├── main_v5.py                # 第五版主程序（多模型+DB+Redis）
│   ├── database.py               # 数据库连接管理（SQLAlchemy）
│   ├── models.py                 # 数据模型（Prediction表结构）
│   └── app.log                   # 结构化日志文件（自动生成）
├── streamlit_app.py              # 前端交互界面（Streamlit）
├── export_onnx.py                # ONNX模型导出脚本（含TorchDynamo优化）
├── export_torchscript.py         # TorchScript模型导出脚本
├── requirements.txt              # 项目依赖清单
├── Dockerfile                    # 容器化配置文件
├── .gitignore                    # Git忽略规则（避免上传冗余文件）
├── fashion_model.pth             # 原始PyTorch模型
├── fashion_model_scripted.pt     # TorchScript优化模型
├── fashion_model.onnx            # ONNX跨框架模型
├── test.db                       # SQLite数据库文件（自动生成）
└── README.md                     # 项目说明文档
🎯 核心功能
1. 多模型图像分类
支持 PNG/JPG/JPEG 格式图片上传
自动完成图像预处理（灰度化、28x28 尺寸调整、对比度优化）
可选 4 种推理模型，返回分类结果及置信度（保留 4 位小数）
2. 安全认证
所有 API 请求需携带X-API-Key请求头
支持多密钥配置（如123456、fashion_api_key_789）
非法请求返回 401 错误，日志记录攻击尝试
3. 可观测性能力
日志监控：记录请求 IP、方法、路径、耗时、响应状态码、预测结果
统计查询：/stats接口返回总预测次数（Redis 缓存）
数据持久化：所有预测记录存入 SQLite，支持后续查询分析
4. 性能基准测试
访问/predict-benchmark接口，上传图片即可对比 4 种模型推理速度
返回结果包含：单模型耗时（毫秒）、相对加速比、最快模型推荐
5. 工程化部署
Docker 一键构建镜像：
bash
运行
docker build -t fashion-mnist-api .
容器化运行：
bash
运行
docker run -p 8000:8000 -p 8501:8501 fashion-mnist-api
🔌 API 接口文档
启动后端服务后，访问http://localhost:8000/docs查看交互式 API 文档，核心接口如下：
1. 预测接口（需认证）
POST /predict：PyTorch 原始模型预测
POST /predict-scripted：TorchScript 优化模型预测
POST /predict-onnx：ONNX Runtime 跨框架模型预测
POST /predict-quantized：动态量化模型预测
请求参数：
Headers：X-API-Key: your_secure_key
Body：multipart/form-data格式，包含图片文件
响应示例：
json
{
  "model": "ONNX",
  "class": "T-shirt/top",
  "confidence": 0.9876,
  "inference_time_ms": 2.34
}
2. 基准测试接口（需认证）
POST /predict-benchmark：对比 4 种模型推理速度
响应示例：
json
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
3. 统计接口（需认证）
GET /stats：获取总预测次数
响应示例：
json
{
  "total_predictions": 1234,
  "message": "统计数据来源于Redis缓存"
}
🔮 未来规划
支持批量图片预测与压缩包上传（ZIP/RAR）
增加模型性能监控面板（响应时间、准确率统计可视化）
实现用户历史预测记录查询与结果导出（CSV/Excel）
集成模型再训练流水线，支持自定义数据集更新模型
部署到 Kubernetes 集群，支持自动扩缩容与高可用
🤝 贡献指南
Fork 本仓库
创建特性分支：git checkout -b feature/AmazingFeature
提交更改：git commit -m 'Add some AmazingFeature'
推送到分支：git push origin feature/AmazingFeature
开启 Pull Request
📄 许可证
本项目采用 Apache 2.0 许可证 - 开源可商用，查看 LICENSE 文件了解详情
📞 联系我们
📧 邮箱：1757952556@qq.com
💻 GitHub：lll222qs
🎯 项目地址：https://github.com/lll222qs/ai_project
如果这个项目对你有帮助，请给个⭐️星标支持！