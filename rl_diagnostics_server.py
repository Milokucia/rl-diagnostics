#!/usr/bin/env python3
"""
RL Training Diagnostics Agent
Supports: PPO, SAC, TD3, DDPG, DQN, + custom algorithms via baselines registry.

Usage:
    pip install tensorboard anthropic flask flask-cors
    python rl_diagnostics_server.py --port 7842
"""

import argparse
import json
import os
from collections import defaultdict
from flask import Flask, jsonify, request
from flask_cors import CORS
import anthropic

app = Flask(__name__)
CORS(app)

client = anthropic.Anthropic()


# ---------------------------------------------------------------------------
# Baseline registry
# Each entry defines:
#   tags        - canonical TensorBoard tag patterns (substring match)
#   failures    - known failure modes with detection heuristics (for prompt context)
#   healthy     - what healthy training looks like for this algo
#   custom      - whether this is a user-defined algo
# ---------------------------------------------------------------------------

BASELINES: dict[str, dict] = {
    "ppo": {
        "name": "Proximal Policy Optimization (PPO)",
        "family": "on-policy actor-critic",
        "tags": {
            "reward":       ["rollout/ep_rew_mean", "train/reward"],
            "policy_loss":  ["train/policy_gradient_loss", "train/pg_loss"],
            "value_loss":   ["train/value_loss"],
            "entropy":      ["train/entropy_loss"],
            "kl":           ["train/approx_kl"],
            "clip_frac":    ["train/clip_fraction"],
            "explained_var":["train/explained_variance"],
            "lr":           ["train/learning_rate"],
        },
        "failures": [
            "Entropy collapse: entropy_loss approaches 0, policy becomes deterministic too early → increase ent_coef",
            "KL spike: approx_kl >> clip_range (typically >0.2) → reduce learning rate or tighten clip_range",
            "Value overfit: value_loss keeps rising while policy_loss plateaus → increase vf_coef or add value head regularization",
            "Reward plateau: ep_rew_mean flat for >20% of total steps → likely reward shaping issue or local optimum",
            "Explained variance negative: value function worse than mean baseline → increase n_epochs or reduce learning rate",
            "Clip fraction saturated: clip_fraction consistently >0.3 → steps too large, reduce learning rate",
        ],
        "healthy": "ep_rew_mean increasing, entropy_loss slowly decreasing, approx_kl <0.02, explained_variance >0.8, clip_fraction 0.05-0.2",
    },
    "sac": {
        "name": "Soft Actor-Critic (SAC)",
        "family": "off-policy actor-critic (continuous)",
        "tags": {
            "reward":       ["rollout/ep_rew_mean", "train/reward"],
            "actor_loss":   ["train/actor_loss"],
            "critic_loss":  ["train/critic_loss"],
            "alpha":        ["train/ent_coef", "train/alpha"],
            "alpha_loss":   ["train/ent_coef_loss"],
            "q_value":      ["train/qf1_values", "train/qf2_values"],
        },
        "failures": [
            "Q-value overestimation: q_values diverging upward → increase target update frequency or add Q-value clipping",
            "Alpha collapse: ent_coef → 0 too fast, policy deterministic → tune target_entropy or fix alpha_lr",
            "Critic divergence: critic_loss exploding → reduce learning rate, gradient clipping",
            "Actor loss sign flip: actor_loss going positive (should be negative, maximizing Q) → check reward scaling",
            "Replay buffer cold start: critic_loss high in early steps is normal, flag if still high after 10k steps",
        ],
        "healthy": "ep_rew_mean increasing, critic_loss decreasing to low stable value, ent_coef slowly decreasing, q_values increasing with reward",
    },
    "td3": {
        "name": "Twin Delayed Deep Deterministic (TD3)",
        "family": "off-policy deterministic actor-critic (continuous)",
        "tags": {
            "reward":       ["rollout/ep_rew_mean"],
            "actor_loss":   ["train/actor_loss"],
            "critic_loss":  ["train/critic_loss"],
            "q_value":      ["train/qf1_values", "train/qf2_values"],
        },
        "failures": [
            "Critic overestimation: q_values far above actual returns → increase policy_delay or add noise regularization",
            "Actor loss plateau: policy not improving while critic converges → reduce policy_delay",
            "Q-value divergence: critic_loss exploding → reduce learning rate, check reward scale",
            "Low exploration: reward plateau early in training → increase action noise sigma",
        ],
        "healthy": "critic_loss converges first, then actor_loss decreases, q_values track true returns, ep_rew_mean steady improvement",
    },
    "dqn": {
        "name": "Deep Q-Network (DQN)",
        "family": "off-policy value-based (discrete)",
        "tags": {
            "reward":       ["rollout/ep_rew_mean", "train/reward"],
            "loss":         ["train/loss"],
            "q_value":      ["train/q_values", "train/mean_q"],
            "epsilon":      ["rollout/exploration_rate", "train/epsilon"],
        },
        "failures": [
            "Q-value divergence: q_values exploding → reduce learning rate, increase target_update_interval",
            "Epsilon not decaying: exploration_rate stuck → check epsilon schedule",
            "Loss oscillation: loss not converging → increase batch_size or replay buffer size",
            "Reward variance high: episode returns wildly variable → environment stochasticity or insufficient training",
            "Catastrophic forgetting: reward drops after improvement → increase replay buffer, reduce learning rate",
        ],
        "healthy": "loss decreasing, q_values increasing with reward, epsilon decaying on schedule, ep_rew_mean improving",
    },
    "ddpg": {
        "name": "Deep Deterministic Policy Gradient (DDPG)",
        "family": "off-policy deterministic actor-critic (continuous)",
        "tags": {
            "reward":       ["rollout/ep_rew_mean"],
            "actor_loss":   ["train/actor_loss"],
            "critic_loss":  ["train/critic_loss"],
            "q_value":      ["train/q_values"],
        },
        "failures": [
            "Q overestimation: q_values diverging → DDPG is sensitive; consider switching to TD3",
            "Training instability: critic_loss oscillating → reduce learning rate, add gradient clipping",
            "Deterministic policy collapse: actor converges to single action → check action space normalization",
        ],
        "healthy": "critic converges before actor, q_values track returns, ep_rew_mean steady increase",
    },
    "custom": {
        "name": "Custom Algorithm",
        "family": "unknown — inferred from tags",
        "tags": {},
        "failures": [
            "Without algorithm context, diagnostics are based on general RL heuristics.",
            "Flag any loss that diverges, any metric that collapses to 0 or explodes, any reward that plateaus.",
        ],
        "healthy": "All losses decreasing or stable, reward increasing, no NaN-adjacent values (very large or very small).",
    },
}


def register_custom_algo(name: str, definition: dict) -> None:
    """
    Register a user-defined algorithm at runtime.
    definition keys: name, family, tags (dict), failures (list[str]), healthy (str)
    """
    BASELINES[name.lower()] = {
        "name": definition.get("name", name),
        "family": definition.get("family", "custom"),
        "tags": definition.get("tags", {}),
        "failures": definition.get("failures", []),
        "healthy": definition.get("healthy", ""),
        "custom": True,
    }


def detect_algo(tags: list[str]) -> str:
    """
    Auto-detect algorithm from tag names if not specified.
    Returns key into BASELINES.
    """
    tag_str = " ".join(tags).lower()
    if "approx_kl" in tag_str or "clip_fraction" in tag_str:
        return "ppo"
    if "ent_coef" in tag_str or "alpha_loss" in tag_str:
        return "sac"
    if "exploration_rate" in tag_str or "epsilon" in tag_str:
        return "dqn"
    if "actor_loss" in tag_str and "qf1" not in tag_str:
        return "td3"
    return "custom"


# ---------------------------------------------------------------------------
# TensorBoard loading
# ---------------------------------------------------------------------------

def load_tfevents(logdir: str) -> dict[str, list[tuple[int, float]]]:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    merged: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for root, dirs, files in os.walk(logdir):
        for fname in files:
            if not fname.startswith("events.out.tfevents"):
                continue
            path = os.path.join(root, fname)
            ea = EventAccumulator(path, size_guidance={"scalars": 0})
            ea.Reload()
            for tag in ea.Tags().get("scalars", []):
                for e in ea.Scalars(tag):
                    merged[tag].append((e.step, e.value))
    return {
        tag: sorted(set(pts), key=lambda x: x[0])
        for tag, pts in merged.items()
    }


def downsample(series: list[tuple[int, float]], max_pts: int = 200) -> list[tuple[int, float]]:
    if len(series) <= max_pts:
        return series
    step = len(series) / max_pts
    return [series[int(i * step)] for i in range(max_pts)]


# ---------------------------------------------------------------------------
# Prompt builder — algo-aware
# ---------------------------------------------------------------------------

def build_analysis_prompt(
    metrics: dict[str, list[tuple[int, float]]],
    algo_key: str,
) -> str:
    baseline = BASELINES.get(algo_key, BASELINES["custom"])

    lines = []
    for tag, series in metrics.items():
        ds = downsample(series, 100)
        vals = [v for _, v in ds]
        recent = vals[int(len(vals) * 0.9):]
        recent_delta = recent[-1] - recent[0] if len(recent) > 1 else 0.0
        lines.append(
            f"  {tag}:\n"
            f"    steps: {ds[0][0]} → {ds[-1][0]}\n"
            f"    range: [{min(vals):.4g}, {max(vals):.4g}]\n"
            f"    first: {vals[0]:.4g}  last: {vals[-1]:.4g}\n"
            f"    recent_trend (last 10%): {recent_delta:+.4g}"
        )

    known_failures = "\n".join(f"  - {f}" for f in baseline["failures"])
    metrics_block = "\n".join(lines)

    return f"""You are an RL training diagnostics expert specializing in {baseline['name']} ({baseline['family']}).

ALGORITHM CONTEXT:
  Algorithm: {baseline['name']}
  Family: {baseline['family']}
  What healthy training looks like: {baseline['healthy']}

KNOWN FAILURE MODES FOR THIS ALGORITHM:
{known_failures}

OBSERVED METRICS:
{metrics_block}

Your task:
1. TRAINING SUMMARY (3-5 sentences): overall health, how far along, trajectory.
2. FAILURE MODES detected — cross-reference observed metrics against the known failure modes above. For each: name it, cite specific metric values as evidence, give a concrete fix with parameter names.
3. POSITIVE SIGNALS — what is working correctly.
4. NEXT STEPS — 1-3 specific actions ranked by priority.

If this is a custom algorithm, reason from first principles about what the metric names and trends suggest.

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
}}
"""


def build_batch_prompt(runs: list[dict]) -> str:
    """Prompt for batch scanning multiple runs."""
    run_summaries = []
    for r in runs:
        metrics = r["metrics"]
        lines = []
        for tag, series in metrics.items():
            ds = downsample(series, 30)  # coarser for batch
            vals = [v for _, v in ds]
            recent = vals[int(len(vals) * 0.9):]
            recent_delta = recent[-1] - recent[0] if len(recent) > 1 else 0.0
            lines.append(f"    {tag}: first={vals[0]:.4g} last={vals[-1]:.4g} trend={recent_delta:+.4g}")
        run_summaries.append(f"RUN: {r['name']} (algo: {r['algo']})\n" + "\n".join(lines))

    block = "\n\n".join(run_summaries)
    return f"""You are an RL training diagnostics expert. Below are summaries of multiple training runs.

{block}

For each run, assign:
- status: "healthy" | "plateau" | "diverging" | "collapsed" | "promising"
- top_issue: single most critical problem (or "none")
- action: "keep_training" | "kill" | "tune" | "investigate"
- reward_trajectory: "improving" | "flat" | "declining"

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
      "note": "one sentence"
    }}
  ],
  "batch_summary": "2-3 sentence overview of the whole experiment"
}}
"""


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/api/algos", methods=["GET"])
def list_algos():
    """Return available algorithm baselines."""
    return jsonify({
        key: {
            "name": v["name"],
            "family": v["family"],
            "custom": v.get("custom", False),
        }
        for key, v in BASELINES.items()
    })


@app.route("/api/register_algo", methods=["POST"])
def register_algo():
    """Register a custom algorithm baseline at runtime."""
    data = request.json
    name = data.get("key", "").strip().lower()
    if not name:
        return jsonify({"error": "key required"}), 400
    register_custom_algo(name, data)
    return jsonify({"registered": name, "algo": BASELINES[name]})


@app.route("/api/scan", methods=["POST"])
def scan():
    data = request.json
    logdir = data.get("logdir", "")
    algo_hint = data.get("algo", "").lower().strip()  # optional override

    if not logdir or not os.path.isdir(logdir):
        return jsonify({"error": f"Directory not found: {logdir}"}), 400

    try:
        metrics = load_tfevents(logdir)
    except Exception as e:
        return jsonify({"error": f"Failed to load tfevents: {e}"}), 500

    if not metrics:
        return jsonify({"error": "No scalar metrics found in logdir."}), 400

    algo_key = algo_hint if algo_hint in BASELINES else detect_algo(list(metrics.keys()))

    chart_data = {
        tag: [{"step": s, "value": v} for s, v in downsample(series, 150)]
        for tag, series in metrics.items()
    }

    prompt = build_analysis_prompt(metrics, algo_key)
    try:
        msg = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = msg.content[0].text.strip()
        if raw.startswith("```"):
            raw = "\n".join(raw.split("\n")[1:])
            if raw.endswith("```"):
                raw = raw[:-3].rstrip()
        analysis = json.loads(raw)
    except json.JSONDecodeError:
        return jsonify({"error": f"Claude returned non-JSON: {raw[:300]}"}), 500
    except Exception as e:
        return jsonify({"error": f"Claude API error: {e}"}), 500

    return jsonify({
        "algo": algo_key,
        "tags": list(metrics.keys()),
        "chart_data": chart_data,
        "analysis": analysis,
    })


@app.route("/api/batch_scan", methods=["POST"])
def batch_scan():
    """Scan multiple run directories under a parent dir."""
    data = request.json
    parent = data.get("parent_dir", "")
    algo_hint = data.get("algo", "").lower().strip()

    if not parent or not os.path.isdir(parent):
        return jsonify({"error": f"Directory not found: {parent}"}), 400

    # Find immediate subdirectories that contain tfevents
    run_dirs = []
    for entry in sorted(os.scandir(parent), key=lambda e: e.name):
        if not entry.is_dir():
            continue
        has_events = any(
            f.startswith("events.out.tfevents")
            for _, _, files in os.walk(entry.path)
            for f in files
        )
        if has_events:
            run_dirs.append(entry)

    if not run_dirs:
        return jsonify({"error": "No run subdirectories with tfevents found."}), 400

    runs = []
    chart_data = {}
    for entry in run_dirs:
        try:
            metrics = load_tfevents(entry.path)
        except Exception:
            continue
        if not metrics:
            continue
        algo_key = algo_hint if algo_hint in BASELINES else detect_algo(list(metrics.keys()))
        runs.append({"name": entry.name, "algo": algo_key, "metrics": metrics})
        chart_data[entry.name] = {
            tag: [{"step": s, "value": v} for s, v in downsample(series, 80)]
            for tag, series in metrics.items()
        }

    if not runs:
        return jsonify({"error": "Could not load metrics from any run."}), 400

    prompt = build_batch_prompt(runs)
    try:
        msg = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = msg.content[0].text.strip()
        if raw.startswith("```"):
            raw = "\n".join(raw.split("\n")[1:])
            if raw.endswith("```"):
                raw = raw[:-3].rstrip()
        batch_analysis = json.loads(raw)
    except Exception as e:
        return jsonify({"error": f"Claude API error: {e}"}), 500

    return jsonify({
        "runs": [r["name"] for r in runs],
        "chart_data": chart_data,
        "analysis": batch_analysis,
    })


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.json
    history = data.get("history", [])
    context = data.get("context", "")
    question = data.get("question", "")
    algo = data.get("algo", "unknown")

    messages = [
        {"role": "user", "content": f"Training analysis context ({algo}):\n{context}\n\nQuestion: {question}"}
    ]
    if history:
        messages = history + [messages[-1]]

    try:
        msg = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system=f"You are an RL training expert specializing in {algo.upper()}. Answer questions about the training run based on the provided context. Be specific and concise.",
            messages=messages,
        )
        return jsonify({"reply": msg.content[0].text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7842)
    args = parser.parse_args()
    print(f"RL Diagnostics server on http://localhost:{args.port}")
    print(f"Supported algorithms: {', '.join(BASELINES.keys())}")
    app.run(port=args.port, debug=False)
