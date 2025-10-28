
FROM registry.cn-hangzhou.aliyuncs.com/common/python:3.9-slim

WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple/ -r requirements.txt
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]