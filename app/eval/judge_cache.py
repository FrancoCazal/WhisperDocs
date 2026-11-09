"""The verdicts the judge already gave (audit E7).

The judge reads the same answer to the same question many times: a rerun
of a config that answers alike, the base and the head of the prompt gate
on the items a pull request did not change. Asked again, it can rule the
other way, and that flip is the judge's noise, not the system's. The cache
keeps each verdict under a hash of everything the judge read and ran with
(app/eval/judges.py), so the same input gets the same verdict, and costs
no call.

A JSON line per verdict, appended. A failed call is no verdict and is not
kept. A line that does not parse, such as one cut short by two runs
writing at once, is skipped.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from app.config import settings
from app.eval.schemas import JudgeResult


class JudgeCache:
    def __init__(self, path: Path):
        self.path = path
        self._verdicts: dict[str, JudgeResult] | None = None

    def get(self, key: str) -> JudgeResult | None:
        return self._load().get(key)

    def put(self, key: str, verdict: JudgeResult) -> None:
        if verdict.error:
            return
        self._load()[key] = verdict
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps({"key": key, **verdict.model_dump(exclude={"error"})})
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def _load(self) -> dict[str, JudgeResult]:
        if self._verdicts is None:
            self._verdicts = {}
            if self.path.is_file():
                for line in self.path.read_text(encoding="utf-8").splitlines():
                    try:
                        row = json.loads(line)
                        self._verdicts[row.pop("key")] = JudgeResult.model_validate(row)
                    except (ValueError, KeyError, TypeError, AttributeError, ValidationError):
                        continue
        return self._verdicts


def judge_cache_from_settings() -> JudgeCache | None:
    """The cache at AGENT_EVAL_JUDGE_CACHE; None when it is empty."""
    path = settings.agent.eval_judge_cache.strip()
    return JudgeCache(Path(path)) if path else None
