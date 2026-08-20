"""
research_tracker.py

Automated Research Experiment & Iteration Tracker:
- Maintains structured research_log.yaml and RESEARCH_CHANGELOG.md.
- Enforces standardized immutable header reminding the AI agent to log every major
  architectural shift, memory module change, hyperparameter adjustment, performance metric,
  winrate against baselines, move quality, decision decisiveness, and bug fixes.
"""

from typing import Any, Dict, List, Optional
import os
import yaml
import time
from datetime import datetime

SYSTEM_HEADER = {
    "project": "Game of the Generals (Salpakan) Vectorized PyTorch RL",
    "version": "2.0.0",
    "hardware_target": "NVIDIA CUDA (RTX 4080 16GB / RTX GPUs)",
    "architecture": "73-Token Pre-LN Transformer Encoder (d_model=192, nhead=6, layers=6, d_ff=512)",
    "belief_mechanism": "Masked Log-Space Sinkhorn-Knopp Doubly-Stochastic Optimal Transport (10 iters)",
    "optimization": "GAE-PPO (gamma=0.99, lambda=0.95, clip=0.2) with Native AMP bfloat16",
    "self_play_strategy": "Prioritized Fictitious Self-Play (PFSP) with 50-checkpoint pool (w_i=(1-win_rate)^1.5 + 0.05)",
    "protocol_reminder": (
        "AUTOMATIC RESEARCH PROTOCOL: This log MUST be updated after every major architectural "
        "shift (e.g. attention mechanism, memory module, positional encoding, Sinkhorn formulation) "
        "or hyperparameter sweep. Record throughput (SPS), winrate vs baselines, move quality, "
        "decisive decision making, bugs fixed, and existing limitations."
    ),
}


class ResearchTracker:
    """
    Automates logging and markdown report generation for research iterations.
    """

    def __init__(
        self,
        yaml_path: str = "research_log.yaml",
        markdown_path: str = "RESEARCH_CHANGELOG.md",
    ) -> None:
        self.yaml_path = yaml_path
        self.markdown_path = markdown_path
        self._ensure_file_exists()

    def _ensure_file_exists(self) -> None:
        if not os.path.isfile(self.yaml_path):
            initial_data = {
                "system_header": SYSTEM_HEADER,
                "experiments": [],
            }
            with open(self.yaml_path, "w", encoding="utf-8") as f:
                yaml.dump(initial_data, f, sort_keys=False, indent=2)
            self._sync_markdown(initial_data)

    def log_experiment(
        self,
        experiment_id: str,
        changes_description: str,
        metrics: Dict[str, Any],
        speed_sps: float,
        winrate_vs_heuristic: float,
        decision_quality: str,
        bugs_fixed: str,
        limitations: str,
        status: str = "COMPLETED",
    ) -> None:
        """
        Appends an experimental result entry to research_log.yaml and syncs markdown changelog.
        """
        with open(self.yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        data["system_header"] = SYSTEM_HEADER
        if "experiments" not in data:
            data["experiments"] = []

        entry = {
            "id": experiment_id,
            "timestamp": datetime.now().isoformat(),
            "changes": changes_description,
            "speed_sps": round(float(speed_sps), 1),
            "winrate_vs_heuristic": round(float(winrate_vs_heuristic), 3),
            "metrics": {
                "loss_policy": round(float(metrics.get("loss_policy", 0.0)), 4),
                "loss_value": round(float(metrics.get("loss_value", 0.0)), 4),
                "loss_sinkhorn": round(float(metrics.get("loss_sinkhorn", 0.0)), 4),
                "loss_entropy": round(float(metrics.get("loss_entropy", 0.0)), 3),
                "combats_per_rollout": round(float(metrics.get("combats_per_rollout", 0.0)), 1),
            },
            "decision_quality": decision_quality,
            "bugs_fixed": bugs_fixed,
            "limitations": limitations,
            "status": status,
        }

        # Check if entry already exists (update it) or append
        existing_idx = next((i for i, e in enumerate(data["experiments"]) if e["id"] == experiment_id), -1)
        if existing_idx >= 0:
            data["experiments"][existing_idx] = entry
        else:
            data["experiments"].append(entry)

        with open(self.yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, sort_keys=False, indent=2)

        self._sync_markdown(data)
        print(f"[ResearchTracker] Logged experiment '{experiment_id}' to {self.yaml_path} & {self.markdown_path}")

    def _sync_markdown(self, data: Dict[str, Any]) -> None:
        """Generates comprehensive markdown changelog from YAML data."""
        header = data.get("system_header", SYSTEM_HEADER)
        exps = data.get("experiments", [])

        md = []
        md.append("# 🧪 Game of the Generals (Salpakan) - Research & Experiment Log\n")
        md.append("> **System Header & Architecture Specification**")
        md.append(f"> - **Project**: {header.get('project')}")
        md.append(f"> - **Architecture**: `{header.get('architecture')}`")
        md.append(f"> - **Belief Module**: {header.get('belief_mechanism')}")
        md.append(f"> - **Self-Play**: {header.get('self_play_strategy')}")
        md.append(f"> - **Optimization**: {header.get('optimization')}")
        md.append(f">\n> 📌 **Protocol Reminder**: *{header.get('protocol_reminder')}*\n")
        md.append("---\n")
        md.append("## 📊 Historical Experiment Trajectory\n")

        if not exps:
            md.append("*No experiment runs logged yet.*")
        else:
            for exp in reversed(exps):
                m = exp.get("metrics", {})
                md.append(f"### 🔬 `{exp.get('id')}` — {exp.get('timestamp')[:19]}")
                md.append(f"- **Status**: `{exp.get('status')}`")
                md.append(f"- **Architectural / Algorithmic Change**: {exp.get('changes')}")
                md.append(f"- **Throughput**: `{exp.get('speed_sps')} SPS`")
                md.append(f"- **Winrate vs Heuristic**: `{(exp.get('winrate_vs_heuristic', 0.0) * 100):.1f}%`")
                md.append(f"- **Losses**: $L_{{pol}}={m.get('loss_policy')}$, $L_{{val}}={m.get('loss_value')}$, $L_{{sink}}={m.get('loss_sinkhorn')}$, $\\text{{Ent}}={m.get('loss_entropy')}$")
                md.append(f"- **Decision Quality Assessment**: {exp.get('decision_quality')}")
                md.append(f"- **Bugs Resolved**: {exp.get('bugs_fixed')}")
                md.append(f"- **Known Limitations**: {exp.get('limitations')}")
                md.append("\n---\n")

        with open(self.markdown_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md))


if __name__ == "__main__":
    tracker = ResearchTracker()
    tracker.log_experiment(
        experiment_id="EXP-001-VECTORIZED-CUDA-PFSP",
        changes_description="Vectorized CUDA Tensor Environment, 73-token Pre-LN Transformer, Masked Sinkhorn-Knopp, and PFSP Opponent Pool",
        metrics={
            "loss_policy": 0.0354,
            "loss_value": 0.0099,
            "loss_sinkhorn": 2.3383,
            "loss_entropy": 2.910,
            "combats_per_rollout": 1332.0,
        },
        speed_sps=105.0,
        winrate_vs_heuristic=0.54,
        decision_quality="Aggressive initial probe with Privates and Spies; Flag retreats safely away from combat zones.",
        bugs_fixed="Fixed canonical direction inversion (dir ^ 1) and bfloat16 index_put tensor assignment.",
        limitations="Log-space Sinkhorn iterations on GPU have fixed 10-step horizon; mobile GPU power throttling during large batch backprop.",
        status="COMPLETED",
    )
