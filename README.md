# AI Agent Engineer 30 Days

## 1. 项目目标
30 天完成从后端工程到 AI Agent 工程的能力升级，产出 2-3 个可演示项目。

## 2. Day 1 交付
- [x] Python 3.11 + 虚拟环境
- [x] FastAPI 项目骨架
- [x] `/health` 接口可用
- [x] 依赖清单与环境变量模板

## 3. 快速开始
```bash
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload