import torch
from main_v2 import FashionCNN  # 注意：将 'your_fastapi_file_name' 替换为你的 FastAPI 文件名（不含 .py）

# 1. 实例化模型并加载权重
model = FashionCNN()
model.load_state_dict(torch.load("fashion_model.pth", map_location=torch.device('cpu')))
model.eval() # 确保模型处于评估模式

# 2. 创建一个示例输入。对于 Fashion-MNIST，输入是 (batch_size, 1, 28, 28)
#    这个示例输入只是为了告诉 TorchScript 模型期望的输入形状和类型。
example_input = torch.randn(1, 1, 28, 28)

# 3. 使用 torch.jit.trace 导出模型
#    它会执行一次模型，并记录下计算图。
traced_script_module = torch.jit.trace(model, example_input)

# 4. 保存导出的模型
traced_script_module.save("fashion_model_scripted.pt")

print("模型已成功导出为 TorchScript 格式: fashion_model_scripted.pt")