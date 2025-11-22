import torch
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import time

# --- 1. 准备数据 ---
def prepare_data():
    print("正在准备CIFAR-10测试数据...")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # ResNet-18需要224x224输入
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # 关键修改：
    # - root='./data'：指定数据集根目录，需包含 cifar-10-batches-py 文件夹
    # - download=False：已手动放置数据，无需下载
    test_dataset = datasets.CIFAR10(
        root='./data',
        train=False,
        download=False,
        transform=transform
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0  # Windows建议设为0，避免多进程错误
    )
    
    print("数据准备完毕！")
    return test_loader

# --- 2. 定义基准测试函数 ---
def benchmark(model, data_loader, device='cpu'):
    model.to(device)
    model.eval()  # 切换到评估模式
    total_time = 0.0
    
    # 关闭梯度计算，加速推理
    with torch.no_grad():
        for images, _ in data_loader:
            images = images.to(device)
            start_time = time.time()
            outputs = model(images)  # 推理
            end_time = time.time()
            total_time += (end_time - start_time)
    
    # 计算每个样本的平均推理时间（毫秒）
    avg_time_per_sample = (total_time / len(data_loader.dataset)) * 1000
    return avg_time_per_sample

# --- 3. 主函数 ---
def main():
    # 步骤1：准备数据
    test_loader = prepare_data()
    
    # 步骤2：加载ResNet-18模型（适配CIFAR-10的10个类别）
    print("\n正在加载ResNet-18模型...")
    model = models.resnet18(pretrained=False)  # 使用随机初始化权重（仅测试速度）
    num_ftrs = model.fc.in_features  # 获取全连接层输入特征数
    model.fc = torch.nn.Linear(num_ftrs, 10)  # 修改输出层为10类
    model.eval()
    print("模型加载完毕！")

    # 步骤3：CPU上性能测试
    print("\n--- 开始在CPU上进行性能测试 ---")
    
    # 测试原始PyTorch模型
    print("\n正在测试原始PyTorch模型...")
    original_model_time = benchmark(model, test_loader, device='cpu')
    print(f"原始PyTorch模型 - 平均每个样本推理时间: {original_model_time:.4f} ms")
    
    # 转换为TorchScript模型
    print("\n正在将模型转换为TorchScript...")
    scripted_model = torch.jit.script(model)  # 脚本化转换
    print("转换完成！")
    
    # 测试TorchScript模型
    print("\n正在测试TorchScript模型...")
    scripted_model_time = benchmark(scripted_model, test_loader, device='cpu')
    print(f"TorchScript模型 - 平均每个样本推理时间: {scripted_model_time:.4f} ms")
    
    # 步骤4：计算加速比并输出结果
    speedup = original_model_time / scripted_model_time
    print("\n" + "="*50)
    print("--- 性能对比结果 ---")
    print(f"原始模型耗时: {original_model_time:.4f} ms")
    print(f"TorchScript模型耗时: {scripted_model_time:.4f} ms")
    print(f"TorchScript相对原始模型的加速比: {speedup:.2f}x")
    print("="*50)

if __name__ == "__main__":
    main()