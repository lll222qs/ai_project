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
import onnxruntime
import numpy as np

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

    # --- 3. 动态量化 PyTorch 模型 ---
# 注意：动态量化对某些模型（如Transformer）效果显著，但对简单的CNN可能效果有限。
model_quantized = None
try:
    model_quantized = torch.quantization.quantize_dynamic(
        model_pytorch,  # 原始模型
        {torch.nn.Linear},  # 指定要量化的层类型
        dtype=torch.qint8  # 量化目标数据类型
    )
    model_quantized.eval()
    logger.info("动态量化 PyTorch 模型加载成功。")
except Exception as e:
    logger.error(f"创建动态量化模型失败: {e}")
    model_quantized = None

    # 4. 加载 ONNX 模型
    # --- 新增：加载 ONNX 模型 ---
model_onnx = None
try:
    # 创建一个 ONNX Runtime 会话
    session = onnxruntime.InferenceSession("fashion_model.onnx")
    
    # 获取模型的输入和输出名称，以便后续调用
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    model_onnx = {
        "session": session,
        "input_name": input_name,
        "output_name": output_name
    }
    logger.info("ONNX 模型加载成功。")
except Exception as e:
    logger.error(f"加载 ONNX 模型失败: {e}")
    model_onnx = None


# --- 图片预处理函数 ---

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
    endpoints = {
        "/predict": "使用原始 PyTorch 模型进行预测",
        "/predict-scripted": "使用 TorchScript 优化模型进行预测",
        "/predict-quantized": "使用动态量化 PyTorch 模型进行预测 (新增, 可选)",
        "/predict-onnx": "使用 ONNX Runtime 优化模型进行预测 (新增)",
        "/predict-benchmark": "对比四个模型的推理速度",
        "/stats": "获取预测统计信息"
    }
    
    return {
        "message": "Fashion-MNIST API is alive (with DB, Background Tasks & ONNX)",
        "endpoints": endpoints
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
    
    # --- 新增：ONNX 预测的公共函数 ---
def _predict_onnx_common(file: UploadFile, model_onnx_dict, db, background_tasks, r):
    """使用 ONNX Runtime 进行预测的公共逻辑"""
    if not model_onnx_dict:
        raise HTTPException(status_code=503, detail="ONNX 模型未加载，服务暂时不可用。")

    try:
        image = Image.open(file.file)
        input_tensor = preprocess_image(image) # 这仍然返回一个 PyTorch 张量

        # 关键：将 PyTorch 张量转换为 NumPy 数组
        input_numpy = input_tensor.numpy()

        # 使用 ONNX Runtime 进行推理
        session = model_onnx_dict["session"]
        input_name = model_onnx_dict["input_name"]
        output_name = model_onnx_dict["output_name"]
        
        start_time = time.time()
        # 注意：ONNX Runtime 的输入是一个字典 {input_name: input_data}
        outputs = session.run([output_name], {input_name: input_numpy})
        inference_time = time.time() - start_time
        
        # outputs 是一个列表，我们取第一个元素，它是一个 NumPy 数组
        logits_numpy = outputs[0]
        
        # 将 NumPy 数组转换回 PyTorch 张量以便后续处理（或者直接用 NumPy 处理）
        logits = torch.from_numpy(logits_numpy)
        
        probabilities = torch.softmax(logits, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)

        predicted_class_name = class_names[predicted_idx.item()]
        result = {
            "model": "ONNX",
            "class": predicted_class_name,
            "confidence": round(confidence.item(), 4),
            "inference_time_ms": round(inference_time * 1000, 2)
        }

        # --- 以下逻辑与 _predict_common 相同 ---
        logger.info(f"将 ONNX 预测结果存入数据库: {file.filename} -> {predicted_class_name}")
        db_prediction = Prediction(
            filename=file.filename,
            model_used="ONNX",
            predicted_class=predicted_class_name,
            confidence=round(confidence.item(), 4),
            inference_time_ms=round(inference_time * 1000, 2)
        )
        db.add(db_prediction)
        db.commit()
        
        logger.info("调度后台任务：更新总预测次数。")
        background_tasks.add_task(update_prediction_count, r)

        logger.info(f"ONNX 模型预测成功: {file.filename}")
        return JSONResponse(content=result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"ONNX 模型预测失败: {file.filename}, 错误: {e}")
        raise HTTPException(status_code=400, detail=f"ONNX 预测失败: {str(e)}")

# 原始 PyTorch 模型预测接口
@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(), # 8. 注入 BackgroundTasks
    api_key: str = Depends(verify_api_key),
    db: Session = Depends(get_db), # 9. 注入数据库会话
    r: redis.Redis = Depends(get_redis) # 10. 注入 Redis 连接
):
    return _predict_common(file, model_pytorch, "PyTorch", db, background_tasks, r)

# TorchScript 模型预测接口
@app.post("/predict-scripted")
async def predict_scripted(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    api_key: str = Depends(verify_api_key),
    db: Session = Depends(get_db),
    r: redis.Redis = Depends(get_redis)
):
    return _predict_common(file, model_scripted, "TorchScript", db, background_tasks, r)

# 新增：动态量化模型预测接口
@app.post("/predict-quantized")
async def predict_quantized(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    api_key: str = Depends(verify_api_key),
    db: Session = Depends(get_db),
    r: redis.Redis = Depends(get_redis)
):
    # 直接复用 _predict_common，但传入量化后的模型
    return _predict_common(file, model_quantized, "PyTorch (Quantized)", db, background_tasks, r)

# --- 新增：ONNX 模型预测接口 ---
@app.post("/predict-onnx")
async def predict_onnx(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    api_key: str = Depends(verify_api_key),
    db: Session = Depends(get_db),
    r: redis.Redis = Depends(get_redis)
):
    # 调用 ONNX 专用的公共预测函数
    return _predict_onnx_common(file, model_onnx, db, background_tasks, r)

# --- 推理速度对比接口 ---(pytorch, scripted,动态量化,onnx)   
@app.post("/predict-benchmark")
async def predict_benchmark(
    file: UploadFile = File(...),
    api_key: str = Depends(verify_api_key)
):
    # 1. 检查所有必需的模型是否都已加载
    models_to_test = {}
    if model_pytorch:
        models_to_test["PyTorch"] = model_pytorch
    if model_scripted:
        models_to_test["TorchScript"] = model_scripted
    if model_quantized:
        models_to_test["PyTorch (Quantized)"] = model_quantized
    if model_onnx:
        models_to_test["ONNX"] = model_onnx
    
    if not models_to_test:
        raise HTTPException(
            status_code=503, 
            detail="没有任何模型被加载，无法进行基准测试。"
        )

    try:
        # 2. 预处理图片 (只做一次)
        image = Image.open(file.file)
        input_tensor = preprocess_image(image)
        
        # 为 ONNX 准备输入（转换为 NumPy 数组）
        input_numpy = input_tensor.numpy()

        # 3. 定义一个字典来存储每个模型的推理时间
        results = {}

        # --- 4. 依次在每个可用的模型上运行推理并计时 ---

        for model_name, model in models_to_test.items():
            logger.info(f"正在为 {model_name} 模型进行基准测试...")
            start_time = time.time()

            if model_name == "ONNX":
                # ONNX 模型使用 ONNX Runtime 进行推理
                session = model["session"]
                input_name = model["input_name"]
                output_name = model["output_name"]
                # ONNX Runtime 的输入是 NumPy 数组
                outputs_onnx = session.run([output_name], {input_name: input_numpy})
            else:
                # PyTorch, TorchScript, Quantized 模型都使用 PyTorch 的方式推理
                with torch.no_grad():
                    # Quantized 模型的输入需要是量化张量
                    if model_name == "PyTorch (Quantized)":
                        # 将普通张量转换为量化张量
                        # 注意：动态量化模型通常期望的是浮点输入，PyTorch 会自动处理量化
                        # 这里的代码与普通 PyTorch 模型相同
                        pass
                    logits = model(input_tensor)
            
            # 计算并记录耗时
            inference_time = time.time() - start_time
            results[model_name] = round(inference_time * 1000, 4) # 转换为毫秒

        # 5. 确定最慢的模型，用于计算相对加速比
        if results:
            slowest_time = max(results.values())
            slowest_model = max(results, key=results.get)
        else:
            slowest_time = 0
            slowest_model = None

        # 6. 构建最终的响应
        benchmark_result = {
            "filename": file.filename,
            "models_tested": list(models_to_test.keys()),
            "inference_times_ms": results,
            "speed_comparison": {}
        }

        # 计算每个模型相对于最慢模型的加速比
        if slowest_time > 0:
            for model_name, time_ms in results.items():
                if time_ms > 0:
                    speedup = slowest_time / time_ms
                    benchmark_result["speed_comparison"][model_name] = f"{speedup:.2f}x"
                else:
                    benchmark_result["speed_comparison"][model_name] = "极快 (耗时可忽略)"
        
        # 找出最快的模型
        if results:
            fastest_model_name = min(results, key=results.get)
            benchmark_result["fastest_model"] = {
                "name": fastest_model_name,
                "time_ms": results[fastest_model_name]
            }
            if slowest_model:
                benchmark_result["fastest_model"]["speedup_over_slowest"] = benchmark_result["speed_comparison"][fastest_model_name]

        # 7. 记录日志
        logger.info(f"基准测试完成: {file.filename} | 测试模型数: {len(models_to_test)} | 最快模型: {benchmark_result['fastest_model']['name']} ({benchmark_result['fastest_model']['time_ms']}ms)")

        return JSONResponse(content=benchmark_result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"基准测试失败: {file.filename}, 错误: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"基准测试失败: {str(e)}")

