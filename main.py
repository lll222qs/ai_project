from fastapi import FastAPI,UploadFile,File
from fastapi.responses import HTMLResponse 
import torch
from torch import nn
from PIL import Image
import numpy as np
from torchvision.transforms import ToTensor
from fastapi.responses import JSONResponse
from PIL import ImageEnhance


app = FastAPI()


class_names = [
    "T_shirt/top","Trouser","Pullover","Dress","Coat",
    "Sandal","Shirt","Sneaker","Bag","Ankle boot"
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

    def forward(self,x):
        logits = self.cnn_stack(x)
        return logits
    

model = FashionCNN()
model.load_state_dict(torch.load("fashion_model.pth",map_location=torch.device('cpu')))
model.eval()



def preprocess_image(image:Image.Image):

    image = image.convert("L")

    image = image.resize((28,28))

    enhancer = ImageEnhance.Contrast(image) 

    image = enhancer.enhance(0.5)  

    tensor = ToTensor()(image)

    tensor = tensor.unsqueeze(0)

    return tensor


@app.get("/")
async def root():
    return {"message":"Fashion-MNIST API is alive"}



@app.post("/predict")
async def predict(file:UploadFile = File(...)):
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
        return JSONResponse(content=result)


    except Exception as e:
       
        return JSONResponse(content={"error": str(e)}, status_code=400)
       

    
