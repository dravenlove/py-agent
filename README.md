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
- [x] Request ID middleware and structured request logging
- [x] In-memory metrics endpoint (`/metrics`)
- [x] Async load test script (`scripts/load_test.py`)

## Day 4 Delivered
- [x] Tool registry (`embed_text`, `rerank_documents`, `calculator`)
- [x] Rule-based agent runner with execution steps
- [x] `/agent` endpoint (`POST`)
- [x] Local agent CLI (`scripts/agent_cli.py`)
- [x] Agent service tests and endpoint tests

## Day 5 Delivered
- [x] Multi-step agent planning (`rerank_documents -> summarize_text`)
- [x] Session memory with `session_id`
- [x] Ability to reuse previous documents from memory
- [x] Agent CLI support for memory-aware runs

## Day 6 Delivered
- [x] Human-in-the-loop confirmation flow for risky actions
- [x] `clear_session_memory` tool for deleting remembered session state
- [x] Pending / cancelled / completed agent statuses
- [x] Agent CLI support for `--confirm` / `--deny`

## Day 7 Delivered
- [x] Agent run IDs for tracing individual executions
- [x] In-memory audit trail for completed, pending, and cancelled runs
- [x] `/agent/runs` endpoint for inspecting recent agent activity
- [x] Run history CLI for filtered audit inspection

## Day 8 Delivered
- [x] Local agent evaluation dataset under `eval/`
- [x] Eval runner for tool selection, status, and answer checks
- [x] Summary metrics for pass rate and routing accuracy
- [x] Eval tests for dataset loading and scoring logic

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

5. Metrics:
```powershell
Invoke-RestMethod http://127.0.0.1:8000/metrics
```

6. Agent:
```powershell
Invoke-RestMethod `
  -Uri http://127.0.0.1:8000/agent `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"input":"请计算 23 * 7"}'
```

7. Agent with memory:
```powershell
Invoke-RestMethod `
  -Uri http://127.0.0.1:8000/agent `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"input":"帮我找出最相关的文档","documents":["进入设置页点击重置密码","联系客服修改邮箱"],"session_id":"demo-1"}'

Invoke-RestMethod `
  -Uri http://127.0.0.1:8000/agent `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"input":"继续比较这些文档并总结一下","session_id":"demo-1"}'
```

8. Agent with confirmation:
```powershell
Invoke-RestMethod `
  -Uri http://127.0.0.1:8000/agent `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"input":"请清空这个会话的记忆","session_id":"demo-1"}'

Invoke-RestMethod `
  -Uri http://127.0.0.1:8000/agent `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"input":"请清空这个会话的记忆","session_id":"demo-1","confirm":true}'
```

9. Agent run history:
```powershell
Invoke-RestMethod http://127.0.0.1:8000/agent/runs

Invoke-RestMethod "http://127.0.0.1:8000/agent/runs?session_id=demo-1&limit=5"
```

## Local CLI Test
```powershell
python scripts/chat_cli.py --message "给我一个 Day 2 学习建议"
python scripts/embedding_cli.py --text "这是一段需要转换成向量的文本"
python scripts/rerank_cli.py --query "怎么重置密码" --doc "进入设置页点击重置密码" --doc "查看账单与发票" --doc "联系客服修改邮箱"
python scripts/agent_cli.py --input "请计算 23 * 7"
python scripts/agent_cli.py --input "帮我找出最相关的文档" --doc "进入设置页点击重置密码" --doc "联系客服修改邮箱"
python scripts/agent_cli.py --input "帮我找出最相关的文档" --doc "进入设置页点击重置密码" --doc "联系客服修改邮箱" --session-id demo-1
python scripts/agent_cli.py --input "继续比较这些文档并总结一下" --session-id demo-1
python scripts/agent_cli.py --input "请清空这个会话的记忆" --session-id demo-1
python scripts/agent_cli.py --input "请清空这个会话的记忆" --session-id demo-1 --confirm
python scripts/agent_runs_cli.py --limit 10
python scripts/agent_runs_cli.py --session-id demo-1 --limit 5
python scripts/eval_agent.py
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
│  ├─ audit.py
│  ├─ agent_service.py
│  ├─ embedding_client.py
│  ├─ llm_client.py
│  ├─ main.py
│  ├─ memory.py
│  ├─ observability.py
│  ├─ rerank_client.py
│  ├─ tools.py
│  ├─ schemas.py
│  └─ settings.py
├─ tests/
│  ├─ test_agent_service.py
│  └─ test_main.py
├─ scripts/
│  ├─ agent_cli.py
│  ├─ agent_runs_cli.py
│  ├─ chat_cli.py
│  ├─ embedding_cli.py
│  ├─ eval_agent.py
│  ├─ load_test.py
│  └─ rerank_cli.py
├─ prompts/
├─ docs/
├─ eval/
│  └─ agent_eval_cases.json
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
