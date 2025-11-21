import torch
import torch.onnx
from app.main_v4 import FashionCNN # 假设你的 FashionCNN 类在 main.py 中

def export_model_to_onnx():
    # 1. 定义模型和示例输入
    model = FashionCNN()
    
    # 加载你训练好的权重
    try:
        model.load_state_dict(torch.load("fashion_model.pth", map_location=torch.device('cpu')))
        model.eval() # 设置为评估模式
        print("PyTorch 模型加载成功。")
    except Exception as e:
        print(f"加载模型失败: {e}")
        return

    # 创建一个与模型输入维度匹配的示例张量 (batch_size=1, channels=1, height=28, width=28)
    dummy_input = torch.randn(1, 1, 28, 28)

    # 2. 使用 TorchDynamo 导出 ONNX
    # torch.onnx.export 现在支持使用 dynamo 作为 backend，以获得更好的优化
    onnx_model_path = "fashion_model.onnx"
    
    print(f"正在使用 TorchDynamo 导出模型到 {onnx_model_path}...")
    
    # 关键：使用 dynamo=True 启用 TorchDynamo 后端
    torch.onnx.export(
        model,
        dummy_input,
        onnx_model_path,
        input_names=["input"],               # 输入张量的名称
        output_names=["output"],             # 输出张量的名称
        dynamic_axes={"input": {0: "batch_size"}}, # 声明 batch_size 为动态维度
        opset_version=12,
        dynamo=True # 启用 TorchDynamo 优化
    )
    
    print(f"ONNX 模型导出成功！已保存至 {onnx_model_path}")

if __name__ == "__main__":
    export_model_to_onnx()