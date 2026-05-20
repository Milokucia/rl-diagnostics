"""
Flask app factory and route definitions.
"""

import os
from flask import Flask, jsonify, request
from flask_cors import CORS

from rl_diagnostics import baselines
from rl_diagnostics.loader import load_tfevents, downsample
from rl_diagnostics import analyzer


def create_app() -> Flask:
    app = Flask(__name__)
    CORS(app)

    # ── Algorithm registry ────────────────────────────────────────────────────

    @app.get("/api/algos")
    def list_algos():
        return jsonify(baselines.list_all())

    @app.post("/api/register_algo")
    def register_algo():
        data = request.json or {}
        key = data.get("key", "").strip().lower()
        if not key:
            return jsonify({"error": "key required"}), 400
        baselines.register(key, data)
        return jsonify({"registered": key, "algo": baselines.get(key)})

    # ── Single run scan ───────────────────────────────────────────────────────

    @app.post("/api/scan")
    def scan():
        data = request.json or {}
        logdir = data.get("logdir", "").strip()
        algo_hint = data.get("algo", "").lower().strip()

        logdir = os.path.expanduser(logdir)
        if not logdir or not os.path.isdir(logdir):
            return jsonify({"error": f"Directory not found: {logdir}"}), 400

        try:
            metrics = load_tfevents(logdir)
        except Exception as e:
            return jsonify({"error": f"Failed to load tfevents: {e}"}), 500

        if not metrics:
            return jsonify({"error": "No scalar metrics found. Check that tensorboard_log was set during training."}), 400

        algo_key = algo_hint if algo_hint in baselines.list_all() else baselines.detect(list(metrics.keys()))

        chart_data = {
            tag: [{"step": s, "value": v} for s, v in downsample(series, 150)]
            for tag, series in metrics.items()
        }

        try:
            analysis = analyzer.analyse(metrics, algo_key)
        except Exception as e:
            return jsonify({"error": f"Claude API error: {e}"}), 500

        return jsonify({
            "algo": algo_key,
            "tags": list(metrics.keys()),
            "chart_data": chart_data,
            "analysis": analysis,
        })

    # ── Batch scan ────────────────────────────────────────────────────────────

    @app.post("/api/batch_scan")
    def batch_scan():
        data = request.json or {}
        parent = os.path.expanduser(data.get("parent_dir", "").strip())
        algo_hint = data.get("algo", "").lower().strip()

        if not parent or not os.path.isdir(parent):
            return jsonify({"error": f"Directory not found: {parent}"}), 400

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
            algo_key = algo_hint if algo_hint in baselines.list_all() else baselines.detect(list(metrics.keys()))
            runs.append({"name": entry.name, "algo": algo_key, "metrics": metrics})
            chart_data[entry.name] = {
                tag: [{"step": s, "value": v} for s, v in downsample(series, 80)]
                for tag, series in metrics.items()
            }

        if not runs:
            return jsonify({"error": "Could not load metrics from any run."}), 400

        try:
            batch_analysis = analyzer.analyse_batch(runs)
        except Exception as e:
            return jsonify({"error": f"Claude API error: {e}"}), 500

        return jsonify({
            "runs": [r["name"] for r in runs],
            "chart_data": chart_data,
            "analysis": batch_analysis,
        })

    # ── Chat ──────────────────────────────────────────────────────────────────

    @app.post("/api/chat")
    def chat():
        data = request.json or {}
        try:
            reply = analyzer.chat(
                question=data.get("question", ""),
                context=data.get("context", ""),
                algo=data.get("algo", "custom"),
                history=data.get("history", []),
            )
            return jsonify({"reply": reply})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return app
