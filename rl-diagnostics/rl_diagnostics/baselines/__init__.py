"""
Algorithm baseline registry.

Each entry defines the known failure modes, healthy training description,
and canonical tag patterns for a given RL algorithm family.

To add your own algorithm, call register() at runtime or POST /api/register_algo.
"""

from __future__ import annotations

_REGISTRY: dict[str, dict] = {
    "ppo": {
        "name": "Proximal Policy Optimization (PPO)",
        "family": "on-policy actor-critic",
        "failures": [
            "Entropy collapse: entropy_loss → 0, policy deterministic too early → increase ent_coef",
            "KL spike: approx_kl >> clip_range (>0.2) → reduce learning_rate or tighten clip_range",
            "Value overfit: value_loss rising while pg_loss plateaus → increase vf_coef or add regularization",
            "Reward plateau: ep_rew_mean flat for >20% of steps → reward shaping or local optimum",
            "Explained variance negative: value function worse than mean baseline → increase n_epochs or reduce lr",
            "Clip fraction saturated: clip_fraction >0.3 consistently → reduce learning_rate",
        ],
        "healthy": (
            "ep_rew_mean increasing, entropy_loss slowly decreasing, approx_kl <0.02, "
            "explained_variance >0.8, clip_fraction 0.05–0.2"
        ),
        "tag_hints": ["approx_kl", "clip_fraction", "entropy_loss", "policy_gradient_loss"],
    },
    "sac": {
        "name": "Soft Actor-Critic (SAC)",
        "family": "off-policy actor-critic (continuous)",
        "failures": [
            "Q overestimation: q_values diverging upward → increase target update freq or add Q clipping",
            "Alpha collapse: ent_coef → 0 too fast → tune target_entropy or fix alpha_lr",
            "Critic divergence: critic_loss exploding → reduce lr, add gradient clipping",
            "Actor loss sign flip: actor_loss positive (should maximize Q) → check reward scaling",
            "Cold start: high critic_loss in first 10k steps is normal; flag if still high after",
        ],
        "healthy": (
            "ep_rew_mean increasing, critic_loss decreasing to low stable value, "
            "ent_coef slowly decreasing, q_values increasing with reward"
        ),
        "tag_hints": ["ent_coef", "actor_loss", "critic_loss", "qf1_values"],
    },
    "td3": {
        "name": "Twin Delayed Deep Deterministic (TD3)",
        "family": "off-policy deterministic actor-critic (continuous)",
        "failures": [
            "Q overestimation: q_values far above actual returns → increase policy_delay or add noise",
            "Actor plateau: policy not improving while critic converges → reduce policy_delay",
            "Q divergence: critic_loss exploding → reduce lr, check reward scale",
            "Low exploration: reward plateau early → increase action noise sigma",
        ],
        "healthy": (
            "critic converges first, then actor_loss decreases, "
            "q_values track true returns, ep_rew_mean steady improvement"
        ),
        "tag_hints": ["actor_loss", "critic_loss", "qf1_values"],
    },
    "ddpg": {
        "name": "Deep Deterministic Policy Gradient (DDPG)",
        "family": "off-policy deterministic actor-critic (continuous)",
        "failures": [
            "Q overestimation: q_values diverging → DDPG is sensitive; consider switching to TD3",
            "Training instability: critic_loss oscillating → reduce lr, add gradient clipping",
            "Deterministic collapse: actor converges to single action → check action space normalization",
        ],
        "healthy": (
            "critic converges before actor, q_values track returns, ep_rew_mean steady increase"
        ),
        "tag_hints": ["actor_loss", "critic_loss", "q_values"],
    },
    "dqn": {
        "name": "Deep Q-Network (DQN)",
        "family": "off-policy value-based (discrete)",
        "failures": [
            "Q divergence: q_values exploding → reduce lr, increase target_update_interval",
            "Epsilon stuck: exploration_rate not decaying → check epsilon schedule",
            "Loss oscillation: loss not converging → increase batch_size or replay buffer",
            "Catastrophic forgetting: reward drops after improvement → larger replay buffer, lower lr",
        ],
        "healthy": (
            "loss decreasing, q_values increasing with reward, "
            "epsilon decaying on schedule, ep_rew_mean improving"
        ),
        "tag_hints": ["exploration_rate", "epsilon", "q_values"],
    },
    "custom": {
        "name": "Custom Algorithm",
        "family": "unknown — inferred from tags",
        "failures": [
            "Reason from first principles: any loss that diverges, any metric that collapses to 0 or explodes, any reward that plateaus.",
        ],
        "healthy": "All losses decreasing or stable, reward increasing, no NaN-adjacent values.",
        "tag_hints": [],
    },
}


def get(key: str) -> dict:
    return _REGISTRY.get(key.lower(), _REGISTRY["custom"])


def detect(tags: list[str]) -> str:
    """Infer algorithm from tag names."""
    tag_str = " ".join(tags).lower()
    scores: dict[str, int] = {}
    for key, baseline in _REGISTRY.items():
        if key == "custom":
            continue
        scores[key] = sum(1 for hint in baseline["tag_hints"] if hint in tag_str)
    best = max(scores, key=lambda k: scores[k])
    return best if scores[best] > 0 else "custom"


def register(key: str, definition: dict) -> None:
    """Register a custom algorithm baseline."""
    _REGISTRY[key.lower()] = {
        "name": definition.get("name", key),
        "family": definition.get("family", "custom"),
        "failures": definition.get("failures", []),
        "healthy": definition.get("healthy", ""),
        "tag_hints": definition.get("tag_hints", []),
        "custom": True,
    }


def list_all() -> dict[str, dict]:
    return {
        k: {"name": v["name"], "family": v["family"], "custom": v.get("custom", False)}
        for k, v in _REGISTRY.items()
    }
