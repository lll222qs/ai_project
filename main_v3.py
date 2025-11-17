import os
import logging
from logging.handlers import RotatingFileHandler

# --- CPU 并行计算优化配置 ---
# 设置 OpenMP 线程数，控制 PyTorch 的 CPU 核心使用量
# 根据你的 CPU 核心数和需求进行调整，例如 "4", "8", "16" 等
# os.cpu_count() 可以获取逻辑核心数，你可以根据需要设置
DESIRED_THREADS = "1" 
os.environ["OMP_NUM_THREADS"] = DESIRED_THREADS
os.environ["MKL_NUM_THREADS"] = DESIRED_THREADS  # 如果你的 PyTorch 基于 MKL

# 打印日志，确认配置已生效
logging.basicConfig(level=logging.INFO)
logger_initial = logging.getLogger("Initialization")
logger_initial.info(f"CPU 优化配置已设置: OMP_NUM_THREADS={os.environ['OMP_NUM_THREADS']}")

from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, status
from fastapi.security import APIKeyHeader
from fastapi.exceptions import RequestValidationError 
from fastapi.requests import Request
from fastapi.responses import JSONResponse
import torch
from torch import nn
from PIL import Image
from PIL import ImageEnhance
from torchvision.transforms import ToTensor
import time
import json

app = FastAPI(title="Fashion-MNIST 识别 API (TorchScript 优化版)")

#请求数据验证失败异常处理
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error(f"请求数据验证失败: {exc.errors()}")
    
    # 自定义错误响应
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "status": "error",
            "code": status.HTTP_422_UNPROCESSABLE_ENTITY,
            "message": "嘿！伙计，请求数据格式不正确。请检查你的请求是否包含了正确的图片文件。",
            "details": exc.errors() # 可选：返回详细的错误信息，方便调试
        },
    )

# --- 配置 ---

# API Key 验证
VALID_API_KEYS = {"123456", "fashion_api_key_789"}
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(api_key: str = Depends(api_key_header)):
    if not api_key or api_key not in VALID_API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效或缺失的 API 密钥",
            headers={"WWW-Authenticate": "X-API-Key"},
        )
    return api_key

# 日志配置
def setup_logging():
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(log_format)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = RotatingFileHandler(
        "app.log",
        maxBytes=10*1024*1024,
        backupCount=3
    )
    file_handler.setFormatter(formatter)

    logging.basicConfig(
        level=logging.INFO,
        handlers=[console_handler, file_handler]
    )

setup_logging()
logger = logging.getLogger(__name__)

# --- 模型和类别定义 ---

class_names = [
    "T_shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# 定义模型结构（用于加载原始 .pth 模型）
class FashionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn_stack = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(in_features=32*7*7, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=10)
        )

    def forward(self, x):
        logits = self.cnn_stack(x)
        return logits

# --- 加载模型 ---

# 1. 加载原始 PyTorch 模型
try:
    model_pytorch = FashionCNN()
    model_pytorch.load_state_dict(torch.load("fashion_model.pth", map_location=torch.device('cpu')))
    model_pytorch.eval()
    logger.info("原始 PyTorch 模型加载成功。")
except Exception as e:
    logger.error(f"加载原始 PyTorch 模型失败: {e}")
    model_pytorch = None

# 2. 加载 TorchScript 模型
try:
    model_scripted = torch.jit.load("fashion_model_scripted.pt", map_location=torch.device('cpu'))
    model_scripted.eval()
    logger.info("TorchScript 模型加载成功。")
except Exception as e:
    logger.error(f"加载 TorchScript 模型失败: {e}")
    model_scripted = None

# --- 图片预处理函数 ---

def preprocess_image(image: Image.Image):
    try:
        image = image.convert("L")  # 转为灰度图
        image = image.resize((28, 28)) #  resize 到 28x28

        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(0.5) # 增强对比度，使其更接近训练数据

        tensor = ToTensor()(image)
        tensor = tensor.unsqueeze(0) # 添加 batch 维度 (1, 1, 28, 28)
        return tensor
    except Exception as e:
        logger.error(f"图片预处理失败: {e}")
        raise

# --- API 接口 ---

@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(
        f"请求处理完成: {request.method} {request.url.path} | 状态码: {response.status_code} | 耗时: {process_time:.4f}s"
    )
    return response

@app.get("/")
async def root():
    return {
        "message": "Fashion-MNIST API is alive",
        "endpoints": {
            "/predict": "使用原始 PyTorch 模型进行预测",
            "/predict-scripted": "使用 TorchScript 优化模型进行预测",
            "/predict-benchmark": "对比两个模型的推理速度"
        }
    }

# 原始 PyTorch 模型预测接口
@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    api_key: str = Depends(verify_api_key)
):
    if not model_pytorch:
        raise HTTPException(status_code=503, detail="原始 PyTorch 模型未加载，服务暂时不可用。")

    try:
        image = Image.open(file.file)
        input_tensor = preprocess_image(image)

        with torch.no_grad():
            start_time = time.time()
            logits = model_pytorch(input_tensor)
            inference_time = time.time() - start_time

            probabilities = torch.softmax(logits, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)

        result = {
            "model": "PyTorch",
            "class": class_names[predicted_idx.item()],
            "confidence": round(confidence.item(), 4),
            "inference_time_ms": round(inference_time * 1000, 2) # 转换为毫秒
        }

        logger.info(f"PyTorch 模型预测成功: {file.filename} -> {result['class']}")
        return JSONResponse(content=result)

    except HTTPException:
        raise # 重新抛出 HTTPException
    except Exception as e:
        logger.error(f"PyTorch 模型预测失败: {file.filename}, 错误: {e}")
        raise HTTPException(status_code=400, detail=f"预测失败: {str(e)}")

# TorchScript 模型预测接口
@app.post("/predict-scripted")
async def predict_scripted(
    file: UploadFile = File(...),
    api_key: str = Depends(verify_api_key)
):
    if not model_scripted:
        raise HTTPException(status_code=503, detail="TorchScript 模型未加载，服务暂时不可用。")

    try:
        image = Image.open(file.file)
        input_tensor = preprocess_image(image)

        # TorchScript 模型在 eval 模式下，no_grad() 仍然可以加速
        with torch.no_grad():
            start_time = time.time()
            logits = model_scripted(input_tensor)
            inference_time = time.time() - start_time

            probabilities = torch.softmax(logits, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)

        result = {
            "model": "TorchScript",
            "class": class_names[predicted_idx.item()],
            "confidence": round(confidence.item(), 4),
            "inference_time_ms": round(inference_time * 1000, 2)
        }

        logger.info(f"TorchScript 模型预测成功: {file.filename} -> {result['class']}")
        return JSONResponse(content=result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"TorchScript 模型预测失败: {file.filename}, 错误: {e}")
        raise HTTPException(status_code=400, detail=f"预测失败: {str(e)}")

# 速度对比接口
@app.post("/predict-benchmark")
async def predict_benchmark(
    file: UploadFile = File(...),
    api_key: str = Depends(verify_api_key)
):
    if not model_pytorch or not model_scripted:
        raise HTTPException(status_code=503, detail="一个或多个模型未加载，无法进行基准测试。")

    try:
        image = Image.open(file.file)
        input_tensor = preprocess_image(image)
        results = []

        # PyTorch 模型
        with torch.no_grad():
            start_time = time.time()
            logits_pytorch = model_pytorch(input_tensor)
            pytorch_time = time.time() - start_time

        # TorchScript 模型
        with torch.no_grad():
            start_time = time.time()
            logits_scripted = model_scripted(input_tensor)
            scripted_time = time.time() - start_time

        # 确保两个模型的输出一致（可选验证步骤）
        # assert torch.allclose(logits_pytorch, logits_scripted), "两个模型的输出不一致！"

        probabilities = torch.softmax(logits_pytorch, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)

        # --- MODIFICATION START ---
        # 处理除以零的情况
        if scripted_time == 0:
            speedup_factor = "无穷大 (TorchScript 速度极快，耗时可忽略)"
        else:
            speedup_factor = f"{pytorch_time / scripted_time:.2f}x"
        # --- MODIFICATION END ---

        result = {
            "class": class_names[predicted_idx.item()],
            "confidence": round(confidence.item(), 4),
            "benchmark": {
                "PyTorch_time_ms": round(pytorch_time * 1000, 2),
                "TorchScript_time_ms": round(scripted_time * 1000, 2),
                "speedup_factor": speedup_factor
            }
        }

        logger.info(f"基准测试完成: {file.filename}")
        return JSONResponse(content=result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"基准测试失败: {file.filename}, 错误: {e}")
        raise HTTPException(status_code=400, detail=f"基准测试失败: {str(e)}")