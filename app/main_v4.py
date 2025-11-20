# app/main.py
import os
import logging
from logging.handlers import RotatingFileHandler
from contextlib import asynccontextmanager

# --- CPU 并行计算优化配置 ---
DESIRED_THREADS = "1" 
os.environ["OMP_NUM_THREADS"] = DESIRED_THREADS
os.environ["MKL_NUM_THREADS"] = DESIRED_THREADS

logging.basicConfig(level=logging.INFO)
logger_initial = logging.getLogger("Initialization")
logger_initial.info(f"CPU 优化配置已设置: OMP_NUM_THREADS={os.environ['OMP_NUM_THREADS']}")
from sqlalchemy.orm import Session
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, status, BackgroundTasks
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

# 1. 导入数据库相关模块
from .database import SessionLocal, engine
from .models import Base, Prediction

# 2. 导入 Redis 模块
import redis

# --- 数据库和 Redis 连接管理 ---

# 在应用启动时创建数据库表
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 应用启动时执行
    logger.info("应用启动，创建数据库表...")
    Base.metadata.create_all(bind=engine)
    
    # 初始化 Redis 连接池
    global redis_pool
    redis_pool = redis.ConnectionPool(
        host="localhost",  # 你的 Redis 地址
        port=6379,         # 你的 Redis 端口
        db=0,
        decode_responses=True  # 自动将 bytes 解码为 string
    )
    logger.info("Redis 连接池初始化成功。")
    
    yield
    
    # 应用关闭时执行
    logger.info("应用关闭，清理资源...")
    # Redis 连接池会自动管理，这里无需额外操作

app = FastAPI(
    title="Fashion-MNIST 识别 API (完整版)",
    lifespan=lifespan  # 绑定 lifespan 事件
)

# 3. 定义获取数据库会话的依赖项
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# 4. 定义获取 Redis 连接的依赖项
def get_redis():
    r = redis.Redis(connection_pool=redis_pool)
    try:
        yield r
    finally:
        r.close()

# --- 其他原有配置 ---

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

# --- 模型和类别定义 (与之前相同) ---

class_names = [
    "T_shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

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

# --- 加载模型 (与之前相同) ---

# ... (此处省略模型加载代码，与你原来的保持一致)
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

# --- 图片预处理函数 (与之前相同) ---

def preprocess_image(image: Image.Image):
    try:
        image = image.convert("L")
        image = image.resize((28, 28))
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(0.5)
        tensor = ToTensor()(image)
        tensor = tensor.unsqueeze(0)
        return tensor
    except Exception as e:
        logger.error(f"图片预处理失败: {e}")
        raise

# --- 5. 定义后台任务函数 ---

def update_prediction_count(r: redis.Redis):
    """
    后台任务：更新 Redis 中的总预测次数。
    """
    try:
        logger.info("后台任务开始：更新总预测次数...")
        # 使用 incr 原子操作增加计数
        r.incr("total_predictions")
        # 可以在这里添加更多复杂的统计逻辑，比如按模型类型统计
        logger.info("后台任务完成：总预测次数已更新。")
    except Exception as e:
        # 即使后台任务失败，也不能影响主请求
        logger.error(f"后台任务 'update_prediction_count' 执行失败: {e}")

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
        "message": "Fashion-MNIST API is alive (with DB & Background Tasks)",
        "endpoints": {
            "/predict": "使用原始 PyTorch 模型进行预测",
            "/predict-scripted": "使用 TorchScript 优化模型进行预测",
            "/predict-benchmark": "对比两个模型的推理速度",
            "/stats": "获取预测统计信息" # 新增端点
        }
    }

# 新增：获取统计信息的接口
@app.get("/stats")
async def get_stats(r: redis.Redis = Depends(get_redis)):
    """
    获取预测统计信息，如总预测次数。
    """
    total = r.get("total_predictions") or 0
    return {
        "total_predictions": int(total),
        "message": "统计数据来源于 Redis 缓存。"
    }

# --- 修改预测接口 ---

def _predict_common(file: UploadFile, model, model_name: str, db, background_tasks, r):
    """
    预测逻辑的公共函数，避免代码重复。
    """
    if not model:
        raise HTTPException(status_code=503, detail=f"{model_name} 模型未加载，服务暂时不可用。")

    try:
        image = Image.open(file.file)
        input_tensor = preprocess_image(image)

        with torch.no_grad():
            start_time = time.time()
            logits = model(input_tensor)
            inference_time = time.time() - start_time

            probabilities = torch.softmax(logits, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)

        predicted_class_name = class_names[predicted_idx.item()]
        result = {
            "model": model_name,
            "class": predicted_class_name,
            "confidence": round(confidence.item(), 4),
            "inference_time_ms": round(inference_time * 1000, 2)
        }

        # 6. 将结果存入数据库
        logger.info(f"将预测结果存入数据库: {file.filename} -> {predicted_class_name}")
        db_prediction = Prediction(
            filename=file.filename,
            model_used=model_name,
            predicted_class=predicted_class_name,
            confidence=round(confidence.item(), 4),
            inference_time_ms=round(inference_time * 1000, 2)
        )
        db.add(db_prediction)
        db.commit()
        db.refresh(db_prediction) # 可选，用于获取数据库自动生成的 ID 等信息
        
        # 7. 调用后台任务
        logger.info("调度后台任务：更新总预测次数。")
        background_tasks.add_task(update_prediction_count, r)

        logger.info(f"{model_name} 模型预测成功: {file.filename}")
        return JSONResponse(content=result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"{model_name} 模型预测失败: {file.filename}, 错误: {e}")
        raise HTTPException(status_code=400, detail=f"预测失败: {str(e)}")

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(), # 8. 注入 BackgroundTasks
    api_key: str = Depends(verify_api_key),
    db: Session = Depends(get_db), # 9. 注入数据库会话
    r: redis.Redis = Depends(get_redis) # 10. 注入 Redis 连接
):
    return _predict_common(file, model_pytorch, "PyTorch", db, background_tasks, r)

@app.post("/predict-scripted")
async def predict_scripted(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    api_key: str = Depends(verify_api_key),
    db: Session = Depends(get_db),
    r: redis.Redis = Depends(get_redis)
):
    return _predict_common(file, model_scripted, "TorchScript", db, background_tasks, r)

# /predict-benchmark 接口可以保持不变，或者也可以对其进行类似的改造
# 为了简洁，这里保持原样
@app.post("/predict-benchmark")
async def predict_benchmark(
    file: UploadFile = File(...),
    api_key: str = Depends(verify_api_key)
):
    # ... (与你原来的代码相同)
    if not model_pytorch or not model_scripted:
        raise HTTPException(status_code=503, detail="一个或多个模型未加载，无法进行基准测试。")

    try:
        image = Image.open(file.file)
        input_tensor = preprocess_image(image)
        results = []

        with torch.no_grad():
            start_time = time.time()
            logits_pytorch = model_pytorch(input_tensor)
            pytorch_time = time.time() - start_time

        with torch.no_grad():
            start_time = time.time()
            logits_scripted = model_scripted(input_tensor)
            scripted_time = time.time() - start_time

        probabilities = torch.softmax(logits_pytorch, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)

        speedup_factor = "无穷大 (TorchScript 速度极快，耗时可忽略)" if scripted_time == 0 else f"{pytorch_time / scripted_time:.2f}x"

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