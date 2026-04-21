from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass

from app.schemas.route_and_plan import RouteAndPlanResponse
from app.services.prompt_builder import PromptPack


@dataclass(frozen=True)
class GeneratedAnswer:
    answer_text: str
    used_evidence_ids: list[str]
    raw_output: str = ""
    generation_mode: str = "fallback"


def _clip_text(text: str, limit: int = 48) -> str:
    text = " ".join(text.strip().split())
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


class AnswerGenerator:
    version = "answer.gen.v2"

    def generate(
        self,
        plan: RouteAndPlanResponse,
        prompt_pack: PromptPack,
        allow_llm_answer: bool = False,
        max_new_tokens: int = 2048,
        temperature: float = 0.3,
    ) -> GeneratedAnswer:
        if allow_llm_answer:
            try:
                return self._generate_with_llm(
                    prompt_pack=prompt_pack,
                    plan=plan,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                )
            except Exception as e:
                print("DEBUG_LLM_EXCEPTION:", repr(e))

        return self.fallback(plan)

    def _generate_with_llm(
        self,
        prompt_pack: PromptPack,
        plan: RouteAndPlanResponse,
        max_new_tokens: int,
        temperature: float,
    ) -> GeneratedAnswer:
        base_url = os.getenv("VLLM_BASE_URL", "").rstrip("/")
        model = os.getenv("VLLM_MODEL", "").strip()
        timeout_sec = int(os.getenv("ANSWER_TIMEOUT_SEC", "60"))
        use_response_format = os.getenv("ANSWER_USE_RESPONSE_FORMAT", "0") == "1"

        if not base_url or not model:
            raise RuntimeError("VLLM_BASE_URL or VLLM_MODEL is not set")

        payload = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_new_tokens,
            "messages": [
                {"role": "system", "content": prompt_pack.system_prompt},
                {"role": "user", "content": prompt_pack.user_prompt},
            ],
        }

        if use_response_format:
            payload["response_format"] = {"type": "json_object"}

        req = urllib.request.Request(
            url=base_url + "/v1/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"LLM HTTPError: {e.code}, body={body}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"LLM URLError: {e.reason}") from e

        print("DEBUG_LLM_RAW:", raw[:2000])

        data = json.loads(raw)
        content = data["choices"][0]["message"]["content"]
        print("DEBUG_LLM_CONTENT:", repr(content[:1000] if isinstance(content, str) else content))

        parsed = self._extract_json_object(content)

        answer_text = str(parsed.get("answer_text", "")).strip()
        used_evidence_ids = [
            str(x)
            for x in parsed.get("used_evidence_ids", [])
            if str(x) in prompt_pack.allowed_evidence_ids
        ]

        if not answer_text:
            raise RuntimeError("LLM returned empty answer_text")

        # -----------------------------
        # Commentary drift 校验逻辑
        # -----------------------------
        if plan.draft_answer_shell.answer_mode == "commentary":
            allowed_keywords = ["分析", "判断", "评论", "典型片段", "依据"]
            if not any(k in answer_text for k in allowed_keywords):
                print("DEBUG_COMMENTARY_FAIL: 缺少核心关键词")
                return self.fallback(plan)
            if len(used_evidence_ids) < 1:
                print("DEBUG_COMMENTARY_FAIL: 没有引用证据")
                return self.fallback(plan)
            if len(answer_text.split()) < 50:
                print("DEBUG_COMMENTARY_FAIL: 输出过短")
                return self.fallback(plan)

        return GeneratedAnswer(
            answer_text=answer_text,
            used_evidence_ids=used_evidence_ids,
            raw_output=content,
            generation_mode="llm",
        )

    def _extract_json_object(self, text: str) -> dict:
        text = text.strip()

        if text.startswith("```"):
            lines = text.splitlines()
            if len(lines) >= 3:
                text = "\n".join(lines[1:-1]).strip()

        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise RuntimeError("No JSON object found in LLM output")

        obj_text = text[start:end + 1]
        return json.loads(obj_text)

    def fallback(self, plan: RouteAndPlanResponse) -> GeneratedAnswer:
        shell = plan.draft_answer_shell
        mode = shell.answer_mode

        top_evidence = plan.evidence_candidates[:2]
        used_evidence_ids = [ev.evidence_id for ev in top_evidence]
        evidence_notes = [_clip_text(ev.text, 46) for ev in top_evidence]

        body: list[str] = []

        if mode == "teaching":
            body.append("先把这个问题的顺序理清楚。")
            for idx, step in enumerate(shell.body_outline[:3], start=1):
                body.append(f"{idx}. {step}")
            if evidence_notes:
                body.append(f"你可以先抓住这个例子：{evidence_notes[0]}")

        elif mode == "bridge":
            body.append("先别急着把它想得太重。")
            if shell.body_outline:
                body.append(shell.body_outline[0])
            if evidence_notes:
                body.append(f"你可以先这样理解：{evidence_notes[0]}")
            if len(shell.body_outline) > 1:
                body.append(shell.body_outline[1])

        elif mode == "commentary":
            body.append("我先直接给判断。")
            for item in shell.body_outline[:2]:
                body.append(item)
            if evidence_notes:
                body.append(f"一个能支撑这个判断的典型片段是：{evidence_notes[0]}")

        else:
            body.append("咱们先把这个感觉接住。")
            if shell.body_outline:
                body.append(shell.body_outline[0])
            if evidence_notes:
                body.append(f"你先抓这个气口：{evidence_notes[0]}")
            if len(shell.body_outline) > 1:
                body.append(shell.body_outline[1])

        answer_parts = [shell.opener] + body + [shell.closer]
        answer_text = "\n".join([x.strip() for x in answer_parts if x.strip()])

        return GeneratedAnswer(
            answer_text=answer_text,
            used_evidence_ids=used_evidence_ids,
            raw_output="",
            generation_mode="fallback",
        )