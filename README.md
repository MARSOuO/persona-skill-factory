# persona-skill-factory online loop v1

这版不是单文件 demo，而是贴近 `/root/persona-skill-factory` 的真实工程骨架。

## 目录

```text
persona-skill-factory/
  app/
    api/routes.py
    core/config.py
    core/dependencies.py
    domain/models.py
    schemas/route_and_plan.py
    adapters/assets.py
    services/router.py
    services/skill_selector.py
    services/evidence_retriever.py
    services/planner.py
    services/engine.py
    main.py
  assets/
    canonical_skills.v1.yaml
  scripts/
    eval_runner.py
  tests/
    test_smoke.py
  requirements.txt
```

## 1. 启动服务

```bash
cd /root/persona-skill-factory
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## 2. 健康检查

```bash
curl http://127.0.0.1:8000/healthz
```

## 3. 调 /route_and_plan

```bash
curl -X POST http://127.0.0.1:8000/route_and_plan \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "请你一步一步给我讲清楚，为什么这个问题不能直接套模板？",
    "top_k_skills": 2,
    "top_k_evidence": 3
  }'
```

## 4. 跑 smoke test

```bash
pytest -q
```

## 5. 用 eval_dataset.v1.jsonl 跑评测

先启动服务，再执行：

```bash
python scripts/eval_runner.py \
  --dataset /root/persona-skill-factory/data/eval_dataset.v1.jsonl \
  --endpoint http://127.0.0.1:8000/route_and_plan \
  --output eval_outputs/route_and_plan_eval.summary.json
```

`eval_runner.py` 已经尽量兼容以下字段差异：

- query / input / prompt / user_query / question
- task / eval_type / subset / kind
- gold_mode / expected_mode / router_label / mode
- gold_skill_ids / expected_skill_ids / skill_ids / gold_skills
- 也会尝试从 `target` / `labels` / `gold` 嵌套对象中取字段

## 6. 这一版的边界

这版仍然故意停在 planner 输出，不直接生成 final answer：

- predicted_mode
- selected_skills
- evidence_candidates
- draft_answer_shell

这是正确的在线 V1 边界。

## 7. 你接下来应该替换什么

优先替换三样：

1. `assets/canonical_skills.v1.yaml` → 你的真实 canonical assets
2. `EvidenceRetrieverService` → skill-bound evidence 改成真实 unit / pgvector / reranker 检索
3. `RouterService` → heuristic router 改成 hybrid router（低置信度时接 vLLM 分类）

## 8. 最关键的工程原则

先让 planner 可评测，再让 generator 可发挥。

你要先知道错在：
- 路由
- skill 选择
- evidence 绑定
- 还是 shell 规划

而不是直接看一句最终回答好不好。
