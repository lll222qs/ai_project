# import gradio as gr
# import requests
# from io import BytesIO 

# FASTAPI_URL = "http://localhost:8000"

# def predict_fashion(image):
   
#     img_byte_arr = BytesIO()
   
#     image.save(img_byte_arr, format="PNG")
   
#     img_byte_arr.seek(0)
    
#     files = {"file": ("image.png", img_byte_arr, "image/png")}
    
 
#     response = requests.post(f"{FASTAPI_URL}/predict", files=files)
    
#     if response.status_code == 200:
#         return response.json()
#     else:
#         return {"error": response.json().get("error", "预测失败")}


# with gr.Blocks(title="Fashion-MNIST 图像分类") as demo:
#     gr.Markdown("# 时尚单品分类器")
#     with gr.Row():
#         input_image = gr.Image(label="上传服装图片", type="pil")
#         output_result = gr.JSON(label="预测结果")
#     predict_btn = gr.Button("开始预测")
#     predict_btn.click(
#         fn=predict_fashion,
#         inputs=input_image,
#         outputs=output_result
#     )

# if __name__ == "__main__":
#     demo.launch()




import gradio as gr
import requests
from PIL import Image  
from io import BytesIO  

def predict_via_api(image_path): 
    
    try:
       
        with Image.open(image_path) as img:
           
            img_byte_arr = BytesIO()
            img.save(img_byte_arr, format="PNG") 
            img_byte_arr.seek(0)  
        
       
        response = requests.post(
            "http://localhost:8000/predict",
            files={"file": ("image.png", img_byte_arr, "image/png")}
        )
        
        if response.status_code == 200:
            result = response.json()
            return f"{result['class']} (置信度: {result['confidence']:.2f})"
        else:
            
            error_msg = response.json().get("error", "未知错误")
            return f"预测失败: {error_msg}"
    except Exception as e:
        return f"处理图像时出错: {str(e)}"


demo = gr.Interface(
    fn=predict_via_api,
    inputs=gr.Image(type="filepath", label="📷 上传衣物图片"),
    outputs=gr.Textbox(label="🎯 预测结果"),
    title="👕 Fashion-MNIST 智能分类器",
    description="上传一张衣物图片,AI将自动识别其类别(T恤、裤子、包等)",
    examples=[["t_shirt_example.jpg"], ["shoe_example.jpg"]],  
    theme="soft"
)

demo.launch(server_name="0.0.0.0", server_port=7860, share=False)