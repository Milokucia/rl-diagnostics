"""
Prompt construction and Claude API calls.
"""

import json
import os

import anthropic

from rl_diagnostics import baselines
from rl_diagnostics.loader import downsample, summarise

_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic()
    return _client


# ── Single run ────────────────────────────────────────────────────────────────

def build_prompt(metrics: dict, algo_key: str) -> str:
    baseline = baselines.get(algo_key)
    known_failures = "\n".join(f"  - {f}" for f in baseline["failures"])

    lines = []
    for tag, series in metrics.items():
        s = summarise(series)
        lines.append(
            f"  {tag}:\n"
            f"    steps: {s['step_start']} → {s['step_end']}\n"
            f"    range: [{s['min']:.4g}, {s['max']:.4g}]\n"
            f"    first: {s['first']:.4g}  last: {s['last']:.4g}\n"
            f"    recent_trend (last 10%): {s['recent_delta']:+.4g}"
        )

    return f"""You are an RL training diagnostics expert specializing in {baseline['name']} ({baseline['family']}).

ALGORITHM CONTEXT:
  Algorithm: {baseline['name']}
  Family: {baseline['family']}
  Healthy training looks like: {baseline['healthy']}

KNOWN FAILURE MODES FOR THIS ALGORITHM:
{known_failures}

OBSERVED METRICS:
{chr(10).join(lines)}

Your task:
1. TRAINING SUMMARY (3-5 sentences): overall health, trajectory, how far along.
2. FAILURE MODES: cross-reference observed metrics against known failure modes. For each: name, severity (critical/warning/info), specific metric evidence, concrete fix with parameter names.
3. POSITIVE SIGNALS: what is working.
4. NEXT STEPS: 1-3 actions ranked by priority.

Respond in JSON only:
{{
  "algo": "{baseline['name']}",
  "summary": "...",
  "health_score": <0-100>,
  "failures": [
    {{"name": "...", "severity": "critical|warning|info", "evidence": "...", "fix": "..."}}
  ],
  "positives": ["..."],
  "next_steps": ["..."]
}}"""


def analyse(metrics: dict, algo_key: str) -> dict:
    prompt = build_prompt(metrics, algo_key)
    msg = _get_client().messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    )
    return _parse_json(msg.content[0].text)


# ── Batch ─────────────────────────────────────────────────────────────────────

def build_batch_prompt(runs: list[dict]) -> str:
    summaries = []
    for r in runs:
        lines = []
        for tag, series in r["metrics"].items():
            s = summarise(series, pts=30)
            lines.append(f"    {tag}: first={s['first']:.4g} last={s['last']:.4g} trend={s['recent_delta']:+.4g}")
        summaries.append(f"RUN: {r['name']} (algo: {r['algo']})\n" + "\n".join(lines))

    block = "\n\n".join(summaries)
    return f"""You are an RL training diagnostics expert reviewing multiple training runs.

{block}

For each run assign:
- status: "healthy" | "plateau" | "diverging" | "collapsed" | "promising"
- health_score: 0-100
- top_issue: single most critical problem or "none"
- action: "keep_training" | "kill" | "tune" | "investigate"
- reward_trajectory: "improving" | "flat" | "declining"
- note: one sentence

Return JSON only:
{{
  "runs": [
    {{
      "name": "...",
      "status": "...",
      "health_score": <0-100>,
      "top_issue": "...",
      "action": "...",
      "reward_trajectory": "...",
      "note": "..."
    }}
  ],
  "batch_summary": "2-3 sentence overview of the whole experiment"
}}"""


def analyse_batch(runs: list[dict]) -> dict:
    prompt = build_batch_prompt(runs)
    msg = _get_client().messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    )
    return _parse_json(msg.content[0].text)


# ── Chat ──────────────────────────────────────────────────────────────────────

def chat(question: str, context: str, algo: str, history: list[dict]) -> str:
    baseline = baselines.get(algo)
    messages = history + [
        {"role": "user", "content": f"Training analysis context:\n{context}\n\nQuestion: {question}"}
    ]
    msg = _get_client().messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=(
            f"You are an RL training expert specializing in {baseline['name']}. "
            "Answer questions about the training run using the provided context. "
            "Be specific and concise. Cite metric names and values."
        ),
        messages=messages,
    )
    return msg.content[0].text


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:])
        if text.strip().endswith("```"):
            text = text.strip()[:-3].rstrip()
    return json.loads(text)
