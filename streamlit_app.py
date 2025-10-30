import streamlit as st
import requests
from PIL import Image
from io import BytesIO
import os

st.set_page_config(page_title="Fashion-MNIST 分类器", page_icon="👕")

# 配置设置
DEMO_MODE = True  # 设置为False当你有真实后端时
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

def mock_predict(image):
    """演示用的模拟预测函数"""
    return {
        "class": "T-shirt",
        "confidence": 0.92,
        "note": "演示模式 - 后端服务部署后即可真实预测"
    }

def real_predict(image):
    """真实的后端预测函数"""
    try:
        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format="PNG")
        img_byte_arr.seek(0)
        
        files = {"file": ("image.png", img_byte_arr, "image/png")}
        response = requests.post(f"{BACKEND_URL}/predict", files=files, timeout=30)
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": "预测服务暂不可用"}
    except Exception as e:
        return {"error": f"请求失败: {str(e)}"}

st.title("👕 Fashion-MNIST 时尚单品分类器")
st.markdown("上传一张衣物图,AI将自动识别其类别")

# 文件上传
uploaded_file = st.file_uploader("选择图片文件", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    # 显示上传的图片
    image = Image.open(uploaded_file)
    st.image(image, caption="上传的图片", use_column_width=True)
    
    # 预测按钮
    if st.button("🎯 开始预测", type="primary"):
        with st.spinner("AI正在分析中..."):
            if DEMO_MODE:
                result = mock_predict(image)
                st.info("🔧 当前为演示模式，真实预测需后端服务支持")
            else:
                result = real_predict(image)
            
            # 显示结果
            if "error" not in result:
                st.success("预测完成！")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("识别结果", result['class'])
                with col2:
                    st.metric("置信度", f"{result['confidence']:.2f}")
                
                # 显示原始JSON
                with st.expander("查看详细数据"):
                    st.json(result)
            else:
                st.error(f"预测失败: {result['error']}")