# app/database.py
import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# SQLALCHEMY_DATABASE_URL = "postgresql://user:password@postgresserver/db"
# 对于 SQLite，URL 格式是 "sqlite:///./<数据库文件名>.db"
# 注意：`check_same_thread` 是 SQLite 的一个特殊参数，FastAPI 是异步的，
# 但 SQLAlchemy 的核心是同步的，这个参数可以避免线程问题。
SQLALCHEMY_DATABASE_URL = "sqlite:///./predictions.db"

# 创建引擎
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)

# 创建一个会话工厂
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 声明性基类，所有的模型都将继承这个类
Base = declarative_base()