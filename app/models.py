# app/models.py
from sqlalchemy import Column, Integer, String, Float, DateTime
from sqlalchemy.sql import func
from .database import Base

# 定义一个 Prediction 模型，它将对应数据库中的 predictions 表
class Prediction(Base):
    __tablename__ = "predictions"

    # 主键
    id = Column(Integer, primary_key=True, index=True)
    # 图片文件名
    filename = Column(String, index=True)
    # 使用的模型
    model_used = Column(String, index=True)
    # 预测结果
    predicted_class = Column(String, index=True)
    # 置信度
    confidence = Column(Float)
    # 推理时间（毫秒）
    inference_time_ms = Column(Float)
    # 创建时间，默认是当前时间
    created_at = Column(DateTime(timezone=True), server_default=func.now())