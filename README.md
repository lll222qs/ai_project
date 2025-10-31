👕 Fashion-MNIST 智能分类器 | 全栈AI应用
https://static.streamlit.io/badges/streamlit_badge_black_white.svg
https://img.shields.io/badge/Python-3.9+-blue.svg
https://img.shields.io/badge/FastAPI-0.68.0-green.svg
https://img.shields.io/badge/PyTorch-2.0.0-red.svg

一个基于现代Web技术的时尚单品图像分类系统，展示从数据训练到产品部署的完整AI工程化流程。

✨ 项目特色
🎯 精准识别 - 基于深度学习模型，准确识别10类时尚单品

🚀 实时预测 - 毫秒级响应，上传图片即刻获得结果

🎨 友好界面 - 简洁直观的Web操作界面

🔧 标准API - 符合REST规范的接口设计

📱 跨端兼容 - 完美支持桌面和移动设备




🏗 系统架构
text
📱 用户界面 (Streamlit) 
    ↓
🌐 REST API (FastAPI + Uvicorn)
    ↓
🧠 AI引擎 (PyTorch + Torchvision)
    ↓
🖼 图像处理 (Pillow + OpenCV)



🛠 技术栈
领域	技术选型
前端界面	Streamlit, Pillow
后端服务	FastAPI, Uvicorn, Pydantic
AI框架	PyTorch, Torchvision, NumPy
工程化	Docker, Git, pip
部署	Streamlit Cloud, Hugging Face Spaces


🚀 快速开始
在线体验
直接访问我们的在线演示：立即体验



本地运行
bash
# 1. 克隆项目
git clone https://github.com/lll222qs/ai_project.git
cd ai_project

# 2. 安装依赖
pip install -r requirements.txt

# 3. 启动后端服务
uvicorn main:app --reload

# 4. 启动前端界面 (新终端)
streamlit run streamlit_app.py


📁 项目结构
text
ai_project/
├── main.py                 # FastAPI后端主程序
├── streamlit_app.py        # Streamlit前端界面
├── requirements.txt        # 项目依赖
├── Dockerfile             # 容器化配置
├── fashion_model.pth      # 训练好的模型
└── README.md              # 项目说明


🎯 核心功能
图像分类
支持PNG、JPG、JPEG格式

自动图像预处理和归一化

返回分类结果及置信度



API接口
python
# 预测接口
POST /predict
Content-Type: multipart/form-data

# 响应示例
{
    "class": "T-shirt",
    "confidence": 0.92
}


🔮 未来规划
支持批量图片预测

添加模型性能监控

实现用户历史记录

增加模型再训练功能



🤝 贡献指南
我们欢迎任何形式的贡献！请参考以下步骤：

Fork 本仓库

创建特性分支 (git checkout -b feature/AmazingFeature)

提交更改 (git commit -m 'Add some AmazingFeature')

推送到分支 (git push origin feature/AmazingFeature)

开启Pull Request



📄 许可证
本项目采用 Apache 2.0 许可证 - 查看 LICENSE 文件了解详情


📞 联系我们
📧 邮箱：你的邮箱

💻 GitHub: lll222qs

🎯 项目地址：https://github.com/lll222qs/ai_project



如果这个项目对你有帮助，请给个⭐️星标支持！