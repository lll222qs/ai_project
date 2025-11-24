# streamlit_app_v5.py
import streamlit as st
import requests
import pandas as pd
from PIL import Image
from io import BytesIO
import time
import plotly.express as px
import plotly.graph_objects as go

# 页面配置
st.set_page_config(
    page_title="AI图像分类器 - 完全体 V5",
    page_icon="🔮",
    layout="wide"
)

# 后端API地址
API_URL = "http://localhost:8000"
API_KEY = "123456"  # 使用你的有效API密钥

def make_authenticated_request(url, files=None, method='GET'):
    """发送带认证的API请求"""
    headers = {"X-API-Key": API_KEY}
    
    try:
        if method == 'GET':
            response = requests.get(url, headers=headers, timeout=10)
        elif method == 'POST':
            response = requests.post(url, files=files, headers=headers, timeout=30)
        else:
            return None
            
        return response
    except requests.exceptions.RequestException as e:
        st.error(f"网络请求失败: {str(e)}")
        return None

def get_system_info():
    """获取系统信息"""
    response = make_authenticated_request(f"{API_URL}/stats")
    if response and response.status_code == 200:
        return response.json()
    return None

# def get_prediction_history():
#     """获取预测历史记录"""
#     # 由于V5后端目前没有提供历史记录查询接口
#     # 这里先返回空列表，等添加该接口后再实现
#     return []

def main():
    st.title("🔮 AI图像分类器 - 完全体 V5")
    st.markdown("基于FastAPI V5后端的全功能图像分类系统")
    
    # 侧边栏 - 系统信息
    st.sidebar.header("📊 系统信息")
    
    # 实时系统状态
    with st.sidebar.expander("实时状态", expanded=True):
        system_info = get_system_info()
        if system_info:
            st.metric("总预测次数", system_info.get('total_predictions', 0))
            st.metric("系统状态", "🟢 在线")
        else:
            st.metric("系统状态", "🔴 离线")
            st.error("无法连接到后端服务")
    
    # 主界面标签页
    tab1, tab2, tab3, tab4 = st.tabs(["📷 图像预测", "⚡ 性能对比", "📊 预测历史", "ℹ️ 系统信息"])
    
    # 标签页1: 图像预测
    with tab1:
        st.header("🎯 多模型图像预测")
        st.markdown("选择不同的优化模型进行图像分类")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            uploaded_file = st.file_uploader(
                "选择图片文件", 
                type=['png', 'jpg', 'jpeg'],
                help="支持 T-shirt、裤子、包等10类时尚单品识别"
            )
            
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption="上传的图片", use_column_width=True)
        
        with col2:
            st.subheader("模型选择")
            model_option = st.radio(
                "选择预测模型:",
                ["PyTorch 原版", "TorchScript 优化", "量化模型", "ONNX 运行时"],
                help="不同模型在精度和速度上有所差异"
            )
            
            model_endpoints = {
                "PyTorch 原版": "/predict",
                "TorchScript 优化": "/predict-scripted", 
                "量化模型": "/predict-quantized",
                "ONNX 运行时": "/predict-onnx"
            }
            
            if st.button("开始预测", type="primary", use_container_width=True):
                if uploaded_file is not None:
                    with st.spinner(f"🔄 正在使用 {model_option} 进行分析..."):
                        try:
                            # 准备图片数据
                            img_byte_arr = BytesIO()
                            image.save(img_byte_arr, format="PNG")
                            img_byte_arr.seek(0)
                            
                            # 调用选择的模型接口
                            endpoint = model_endpoints[model_option]
                            files = {"file": ("image.png", img_byte_arr, "image/png")}
                            response = make_authenticated_request(
                                f"{API_URL}{endpoint}", 
                                files=files, 
                                method='POST'
                            )
                            
                            if response and response.status_code == 200:
                                result = response.json()
                                
                                # 显示结果
                                st.success("✅ 预测完成！")
                                
                                # 结果卡片
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("识别结果", result['class'])
                                with col2:
                                    st.metric("置信度", f"{result['confidence']:.2%}")
                                with col3:
                                    st.metric("推理时间", f"{result['inference_time_ms']}ms")
                                
                                # 置信度可视化
                                st.subheader("置信度分布")
                                fig = go.Figure(data=[
                                    go.Bar(x=[result['class']], y=[result['confidence']],
                                          marker_color='lightblue')
                                ])
                                fig.update_layout(
                                    title=f"{result['class']} 的置信度",
                                    yaxis_title="置信度",
                                    yaxis_range=[0, 1],
                                    showlegend=False
                                )
                                st.plotly_chart(fig, use_container_width=True)
                                
                            else:
                                st.error("❌ 预测失败，请检查后端服务状态")
                                
                        except Exception as e:
                            st.error(f"❌ 请求失败: {str(e)}")
                else:
                    st.warning("⚠️ 请先上传图片文件")
    
    # 标签页2: 性能对比
    with tab2:
        st.header("⚡ 模型性能对比")
        st.markdown("对比不同模型在相同图片上的推理性能")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            benchmark_file = st.file_uploader(
                "选择测试图片", 
                type=['png', 'jpg', 'jpeg'],
                key="benchmark"
            )
            
            if benchmark_file is not None:
                benchmark_image = Image.open(benchmark_file)
                st.image(benchmark_image, caption="性能测试图片", use_container_width=True)
        
        with col2:
            st.subheader("基准测试配置")
            if st.button("开始性能测试", type="primary", use_container_width=True):
                if benchmark_file is not None:
                    with st.spinner("🔄 正在运行基准测试，这可能需要一些时间..."):
                        try:
                            # 准备图片数据
                            img_byte_arr = BytesIO()
                            benchmark_image.save(img_byte_arr, format="PNG")
                            img_byte_arr.seek(0)
                            
                            # 调用基准测试接口
                            files = {"file": ("benchmark.png", img_byte_arr, "image/png")}
                            response = make_authenticated_request(
                                f"{API_URL}/predict-benchmark", 
                                files=files, 
                                method='POST'
                            )
                            
                            if response and response.status_code == 200:
                                benchmark_result = response.json()
                                
                                st.success("✅ 基准测试完成！")
                                
                                # 显示最快模型
                                fastest = benchmark_result.get('fastest_model', {})
                                st.info(f"🚀 最快模型: **{fastest.get('name', 'N/A')}** "
                                      f"({fastest.get('time_ms', 'N/A')}ms)")
                                
                                # 性能对比图表
                                st.subheader("推理时间对比")
                                times_data = benchmark_result.get('inference_times_ms', {})
                                if times_data:
                                    df_times = pd.DataFrame({
                                        'Model': list(times_data.keys()),
                                        'Time (ms)': list(times_data.values())
                                    })
                                    
                                    fig = px.bar(
                                        df_times, 
                                        x='Model', 
                                        y='Time (ms)',
                                        title="各模型推理时间对比",
                                        color='Time (ms)',
                                        color_continuous_scale='Viridis'
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                # 加速比展示
                                st.subheader("性能加速比")
                                speed_data = benchmark_result.get('speed_comparison', {})
                                if speed_data:
                                    for model, speedup in speed_data.items():
                                        st.write(f"- **{model}**: {speedup}")
                                
                            else:
                                st.error("❌ 基准测试失败")
                                
                        except Exception as e:
                            st.error(f"❌ 基准测试失败: {str(e)}")
                else:
                    st.warning("⚠️ 请先上传测试图片")
    
    # 标签页3: 预测历史
    with tab3:
        st.header("📊 预测历史记录")
        st.markdown("查看历史预测记录和统计信息")
        
        # 这里可以显示从数据库获取的历史记录
        # 需要后端提供对应的接口
        history = get_prediction_history()
        
        if history:
            df_history = pd.DataFrame(history)
            st.dataframe(df_history, use_container_width=True)
            
            # 简单的统计图表
            if not df_history.empty:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("模型使用分布")
                    model_counts = df_history['model_used'].value_counts()
                    fig_pie = px.pie(
                        values=model_counts.values,
                        names=model_counts.index,
                        title="各模型使用比例"
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with col2:
                    st.subheader("类别识别分布")
                    class_counts = df_history['predicted_class'].value_counts()
                    fig_bar = px.bar(
                        x=class_counts.index,
                        y=class_counts.values,
                        title="各类别识别次数",
                        labels={'x': '类别', 'y': '次数'}
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("📝 暂无预测历史记录")
            st.markdown("""
            **待开发功能：**
            - 历史预测记录查询
            - 预测结果统计分析  
            - 模型性能趋势分析
            """)
    
    # 标签页4: 系统信息
    with tab4:
        st.header("ℹ️ 系统信息")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("后端服务状态")
            if system_info:
                st.success("🟢 服务在线")
                st.json(system_info)
            else:
                st.error("🔴 服务离线")
            
            st.subheader("API端点说明")
            endpoints_info = {
                "/predict": "PyTorch 原版模型",
                "/predict-scripted": "TorchScript 优化模型", 
                "/predict-quantized": "量化优化模型",
                "/predict-onnx": "ONNX 运行时模型",
                "/predict-benchmark": "多模型性能对比",
                "/stats": "系统统计信息"
            }
            
            for endpoint, desc in endpoints_info.items():
                st.write(f"`{endpoint}` - {desc}")
        
        with col2:
            st.subheader("支持识别的类别")
            classes = [
                "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
            ]
            
            for i, class_name in enumerate(classes, 1):
                st.write(f"{i}. {class_name}")
            
            st.subheader("技术特性")
            features = [
                "✅ 多模型推理支持",
                "✅ 实时性能对比", 
                "✅ 数据库持久化",
                "✅ Redis缓存统计",
                "✅ 后台任务处理",
                "✅ API密钥认证",
                "✅ 详细日志记录"
            ]
            
            for feature in features:
                st.write(feature)

if __name__ == "__main__":
    main()