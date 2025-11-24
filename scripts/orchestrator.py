import subprocess
import time
import signal
import sys
import os
from pathlib import Path

class ServiceManager:
    def __init__(self):
        self.processes = []
        
    def start_service(self, name, command, cwd=None, env=None, is_task=False):
        """启动一个服务或任务"""
        print(f"[启动] {name}")
        try:
            # 设置环境变量
            service_env = os.environ.copy()
            if env:
                service_env.update(env)
                
            process = subprocess.Popen(
                command,
                shell=True,
                cwd=cwd,
                env=service_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8'  # 强制使用UTF-8编码
            )
            
            # 如果是任务，等待完成；如果是服务，加入监控列表
            if is_task:
                print(f"[执行任务] {name}")
                returncode = process.wait(timeout=30)  # 等待任务完成，最多30秒
                
                # 读取任务输出
                stdout_output, stderr_output = process.communicate()
                if stdout_output:
                    print(f"[输出] {name}: {stdout_output.strip()}")
                if stderr_output:
                    print(f"[错误] {name}: {stderr_output.strip()}")
                    
                if returncode == 0:
                    print(f"[成功] {name} 执行成功")
                    return True
                else:
                    print(f"[失败] {name} 执行失败，退出码: {returncode}")
                    return False
            else:
                self.processes.append((name, process))
                print(f"[完成] {name} 启动完成 (PID: {process.pid})")
                
                # 给服务启动留点时间
                time.sleep(3)
                return True
                
        except subprocess.TimeoutExpired:
            print(f"[超时] {name} 执行超时")
            process.kill()
            return False
        except Exception as e:
            print(f"[失败] 启动 {name} 失败: {e}")
            return False
    
    def stop_services(self):
        """停止所有服务"""
        print("\n[停止] 正在停止所有服务...")
        for name, process in self.processes:
            print(f"[停止] 正在停止 {name} (PID: {process.pid})...")
            process.terminate()
            try:
                process.wait(timeout=5)
                print(f"[完成] {name} 已停止")
            except subprocess.TimeoutExpired:
                process.kill()
                print(f"[强制停止] {name} 被强制终止")
        print("[完成] 所有服务已停止")
    
    def monitor_services(self):
        """监控服务状态"""
        try:
            print("\n[监控] 开始监控服务状态...")
            while True:
                time.sleep(5)
                all_healthy = True
                
                for name, process in self.processes:
                    if process.poll() is not None:
                        print(f"[异常] 服务 {name} 已意外退出，退出码: {process.returncode}")
                        # 读取错误输出
                        stderr_output = process.stderr.read()
                        if stderr_output:
                            print(f"[错误信息] {stderr_output}")
                        all_healthy = False
                        
                if all_healthy:
                    print("[正常] 所有服务运行正常...")
                else:
                    print("[异常] 有服务异常，即将停止所有服务...")
                    return False
                    
        except KeyboardInterrupt:
            print("\n\n[停止] 接收到停止信号")
            self.stop_services()
            return True

def main():
    manager = ServiceManager()
    
    # 服务配置 - 根据你的实际项目调整这些命令
    services = [
        # 一次性任务 - 数据库初始化
        {
            "name": "数据库初始化",
            "command": "python -c \"from app.database import engine; from app.models import Base; Base.metadata.create_all(bind=engine); print('Database tables created successfully')\"",
            "cwd": ".",
            "env": {"PYTHONPATH": "."},
            "is_task": True  # 标记为一次性任务
        },
        # 长期运行的服务
        {
            "name": "FastAPI后端", 
            "command": "uvicorn app.main_v5:app --host 0.0.0.0 --port 8000",
            "cwd": ".",
            "env": {"PYTHONPATH": "."},
            "is_task": False  # 标记为长期服务
        },
        {
            "name": "Streamlit前端",
            "command": "streamlit run streamlit_app_v5.py --server.port 8501 --server.address 0.0.0.0 --server.headless true",
            "cwd": ".",
            "env": {"API_URL": "http://localhost:8000"},
            "is_task": False  # 标记为长期服务
        }
    ]
    
    try:
        print("=" * 50)
        print("AI应用服务编排器启动")
        print("=" * 50)
        
        # 启动所有服务
        for service in services:
            success = manager.start_service(**service)
            if not success:
                print("[失败] 服务启动失败，正在停止所有服务...")
                manager.stop_services()
                sys.exit(1)
                
        print("\n" + "=" * 50)
        print("[成功] 所有服务启动成功！")
        print("[访问信息]")
        print("   - FastAPI后端: http://localhost:8000")
        print("   - API文档: http://localhost:8000/docs")  
        print("   - Streamlit前端: http://localhost:8501")
        print("\n按 Ctrl+C 停止所有服务")
        print("=" * 50)
        
        # 监控服务状态
        manager.monitor_services()
        
    except Exception as e:
        print(f"[错误] 发生错误: {e}")
        manager.stop_services()
        sys.exit(1)

if __name__ == "__main__":
    main()