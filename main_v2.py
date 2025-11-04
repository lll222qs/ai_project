from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, status
from fastapi.security import APIKeyHeader
from fastapi.requests import Request 
from fastapi.responses import JSONResponse
import torch
from torch import nn
from PIL import Image
import numpy as np
from torchvision.transforms import ToTensor
from PIL import ImageEnhance
import time
import logging
from logging.handlers import RotatingFileHandler
import json


app = FastAPI(title="Fashion-MNIST 识别 API")


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



def setup_logging():
    log_format = {
        "time": "%(asctime)s",
        "level": "%(levelname)s",
        "message": "%(message)s",
        "module": "%(module)s"
    }
    formatter = logging.Formatter(json.dumps(log_format))

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



@app.middleware("http")  
async def logging_middleware(request: Request, call_next):
   
    start_time = time.time()
    client_ip = request.client.host
    request_method = request.method
    request_path = request.url.path

    
    response = await call_next(request)

 
    process_time = time.time() - start_time
    logger.info(
        json.dumps({
            "event": "request_processed",
            "client_ip": client_ip,
            "method": request_method,
            "path": request_path,
            "process_time_seconds": round(process_time, 4),
            "status_code": response.status_code
        })
    )
    return response



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

model = FashionCNN()
model.load_state_dict(torch.load("fashion_model.pth", map_location=torch.device('cpu')))
model.eval()

def preprocess_image(image: Image.Image):

    image = image.convert("L")

    image = image.resize((28, 28))

    enhancer = ImageEnhance.Contrast(image)

    image = enhancer.enhance(0.5)

    tensor = ToTensor()(image)

    tensor = tensor.unsqueeze(0)
    
    return tensor




@app.get("/")
async def root():
    return {"message": "Fashion-MNIST API is alive"}

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    api_key: str = Depends(verify_api_key)
):
    try:
        image = Image.open(file.file)
        input_tensor = preprocess_image(image)

        with torch.no_grad():
            logits = model(input_tensor)
            probabilities = torch.softmax(logits, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)

        result = {
            "class": class_names[predicted_idx.item()],
            "confidence": round(confidence.item(), 4)
        }

        logger.info(
            json.dumps({
                "event": "prediction_success",
                "filename": file.filename,
                "predicted_class": result["class"],
                "confidence": result["confidence"]
            })
        )
        return JSONResponse(content=result)

    except Exception as e:
        logger.error(
            json.dumps({
                "event": "prediction_failed",
                "filename": file.filename,
                "error_message": str(e)
            })
        )
        return JSONResponse(content={"error": str(e)}, status_code=400)