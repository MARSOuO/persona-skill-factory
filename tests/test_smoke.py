from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_healthz() -> None:
    response = client.get("/healthz")
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["skills_loaded"] >= 1


def test_route_and_plan() -> None:
    response = client.post(
        "/route_and_plan",
        json={
            "query": "请你一步一步给我讲清楚，为什么这个问题不能直接套模板？",
            "top_k_skills": 2,
            "top_k_evidence": 3,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["predicted_mode"]["label"] == "teaching"
    assert len(body["selected_skills"]) == 2
    assert len(body["evidence_candidates"]) == 3
