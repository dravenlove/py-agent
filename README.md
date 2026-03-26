# AI Agent Engineer 30 Days

30 days of hands-on practice to transition into an AI agent engineer.

## Goal
- Build practical backend + LLM + agent workflow skills.
- Deliver portfolio-ready projects with demos and evaluation reports.

## Day 1 Delivered
- [x] Python 3.11 virtual environment
- [x] FastAPI project scaffold
- [x] `/health` endpoint
- [x] `requirements.txt` and `.env.example`

## Day 2 Delivered
- [x] `/chat` endpoint (`POST`)
- [x] `/embeddings` endpoint (`POST`)
- [x] `/rerank` endpoint (`POST`)
- [x] OpenAI API integration (`responses.create`)
- [x] Local CLI test scripts (`scripts/chat_cli.py`, `scripts/embedding_cli.py`, `scripts/rerank_cli.py`)
- [x] Basic API tests (`tests/test_main.py`)

## Day 3 Delivered
- [x] Async service handlers for `/chat`, `/embeddings`, and `/rerank`
- [x] Timeout config for chat, embeddings, and rerank upstream calls
- [x] Shared retry/backoff helper for transient upstream failures
- [x] Unified upstream error mapping (`401`, `404`, `504`, `502`)
- [x] Async load test script (`scripts/load_test.py`)

## Quick Start
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
# Edit .env and set your real OPENAI_API_KEY
uvicorn app.main:app --reload
```

## Endpoints
1. Health check:
```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

2. Chat:
```powershell
Invoke-RestMethod `
  -Uri http://127.0.0.1:8000/chat `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"message":"你好，请用三句话介绍你自己"}'
```

3. Embeddings:
```powershell
Invoke-RestMethod `
  -Uri http://127.0.0.1:8000/embeddings `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"input":"这是一段需要转换成向量的文本"}'
```

4. Rerank:
```powershell
$body = @{
  query = "怎么重置密码"
  documents = @(
    "进入设置页点击重置密码"
    "查看账单与发票"
    "联系客服修改邮箱"
  )
  top_n = 2
} | ConvertTo-Json -Depth 5

Invoke-RestMethod `
  -Uri http://127.0.0.1:8000/rerank `
  -Method Post `
  -ContentType "application/json" `
  -Body $body
```

## Local CLI Test
```powershell
python scripts/chat_cli.py --message "给我一个 Day 2 学习建议"
python scripts/embedding_cli.py --text "这是一段需要转换成向量的文本"
python scripts/rerank_cli.py --query "怎么重置密码" --doc "进入设置页点击重置密码" --doc "查看账单与发票" --doc "联系客服修改邮箱"
```

## Day 3 Load Test
```powershell
python scripts/load_test.py --endpoint embeddings --count 10 --concurrency 5 --text "Day 3 load test sample"
python scripts/load_test.py --endpoint rerank --count 10 --concurrency 5 --text "怎么重置密码"
```

## Project Structure
```text
py-openclaw/
├─ app/
│  ├─ __init__.py
│  ├─ embedding_client.py
│  ├─ llm_client.py
│  ├─ main.py
│  ├─ rerank_client.py
│  ├─ schemas.py
│  └─ settings.py
├─ tests/
│  └─ test_main.py
├─ scripts/
│  ├─ chat_cli.py
│  ├─ embedding_cli.py
│  ├─ load_test.py
│  └─ rerank_cli.py
├─ prompts/
├─ docs/
├─ eval/
├─ .env.example
├─ .gitignore
├─ requirements.txt
└─ README.md
```

## Run Tests
```powershell
pytest -q
```

## Git Flow (day2 -> main)
```powershell
git switch day1
git add .
git commit -m "feat(day2): add chat endpoint and openai integration"
git push origin day1

git switch main
git pull origin main
git merge --no-ff day1 -m "merge: day1 day2 progress into main"
git push origin main
```
