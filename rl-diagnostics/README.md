# rl-diagnostics

**AI-powered RL training diagnostics. No graphs. Just answers.**

Point `rl-diagnostics` at any TensorBoard logdir and get a structured failure analysis — entropy collapse, value loss explosion, reward plateau, KL divergence spikes — with exact parameter fixes. No graphs. No manual inspection.

Built for researchers running mass parallel experiments who can't eyeball 50 TensorBoard dashboards.

---

## Install

```bash
git clone https://github.com/yourusername/rl-diagnostics
cd rl-diagnostics
pip install -e .
```

## Quickstart

```bash
export ANTHROPIC_API_KEY=sk-...
rl-diagnostics
```

Open the UI, point it at your logdir:

```
~/mk-robotics/dex-hand-sim/logs
```

The server accepts `~` paths — no need for absolute paths.

---

## Supported Algorithms

| Algorithm | Auto-detect | Key failure modes |
|-----------|:-----------:|-------------------|
| PPO       | ✓ | entropy collapse, KL spike, value overfit, clip saturation |
| SAC       | ✓ | Q overestimation, alpha collapse, critic divergence |
| TD3       | ✓ | Q overestimation, actor plateau, low exploration |
| DDPG      | ✓ | training instability, deterministic collapse |
| DQN       | ✓ | Q divergence, epsilon decay failure, catastrophic forgetting |
| Custom    | via API | define your own failure modes at runtime |

Algorithm is auto-detected from tag names. You can also specify it explicitly in the UI.

---

## Custom Algorithm Registration

Register your own algorithm baseline at runtime — no code changes needed:

```bash
curl -X POST http://localhost:7842/api/register_algo \
  -H "Content-Type: application/json" \
  -d '{
    "key": "drq",
    "name": "Data-Regularized Q (DrQ)",
    "family": "off-policy actor-critic with image augmentation",
    "failures": [
      "Augmentation collapse: policy overfits to augmentation artifacts",
      "Encoder divergence: conv features destabilizing early training"
    ],
    "healthy": "critic loss decreasing, encoder features stable, reward improving"
  }'
```

Once registered, select `drq` in the UI and diagnostics are specific to that algorithm's failure modes.

---

## Batch Scan — Mass Experiments

Point it at a parent directory containing multiple run subdirs:

```
POST /api/batch_scan
{ "parent_dir": "~/runs/experiment_01" }
```

Returns a ranked triage table — every run scored 0–100, status (`healthy / plateau / diverging / collapsed`), and a recommended action (`keep_training / kill / tune / investigate`). Click any run to expand its metric sparklines.

No more opening 50 TensorBoard tabs.

---

## API Reference

| Endpoint | Method | Body | Description |
|----------|--------|------|-------------|
| `/api/scan` | POST | `{ logdir, algo? }` | Single run analysis |
| `/api/batch_scan` | POST | `{ parent_dir, algo? }` | Multi-run triage |
| `/api/chat` | POST | `{ question, context, algo, history }` | Follow-up Q&A |
| `/api/algos` | GET | — | List registered algorithms |
| `/api/register_algo` | POST | algorithm definition | Register custom algorithm |

---

## Project Structure

```
rl-diagnostics/
├── rl_diagnostics/
│   ├── cli.py          # entry point: rl-diagnostics command
│   ├── server.py       # Flask app factory + routes
│   ├── loader.py       # TensorBoard event file parsing
│   ├── analyzer.py     # prompt construction + Claude API
│   └── baselines/      # algorithm failure mode registry
│       └── __init__.py
├── frontend/
│   └── RLDiagnosticsAgent.jsx
├── examples/
│   └── sample_logs/    # small tfevents for testing
├── pyproject.toml
└── README.md
```

---

## Why This Exists

TensorBoard is a viewer. It has no diagnostic intelligence. When you're running mass hyperparameter sweeps on a dexterous manipulation task, you cannot manually review every run. This tool closes the gap between raw training metrics and the decision of what to do next.

---

## Contributing

The most valuable contribution is algorithm baselines. See `rl_diagnostics/baselines/__init__.py` — each baseline is a dict with `name`, `family`, `failures` (list of strings), `healthy` (string), and `tag_hints` (list of tag substrings for auto-detection).

PRs adding DrQ, REDQ, DreamerV3, IMPALA, R2D2, TD-MPC2 welcome.

---

## License

MIT
