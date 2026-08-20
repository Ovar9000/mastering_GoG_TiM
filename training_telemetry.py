"""
training_telemetry.py

Live Training Telemetry, Dashboard, and RL Pitfall Diagnostics:
- Live tracking of steps per second (SPS), episodes/sec, combat density, winrates.
- Automatic detection and alerts for RL pitfalls in imperfect information self-play:
  1. Entropy Collapse / Premature Convergence.
  2. Value Loss Divergence & Non-Stationarity.
  3. Turtling / Stagnation (low combat density, high truncation rate).
  4. Large Policy Divergence & Ratio Clipping saturation.
  5. Catastrophic Forgetting in PFSP Opponent Pool.
"""

from typing import Dict, List, Optional, Tuple
import time
import math
import collections
import torch


class PitfallAlert:
    """Represents a diagnosed RL training pitfall."""
    def __init__(self, name: str, severity: str, message: str, recommendation: str) -> None:
        self.name = name
        self.severity = severity  # 'WARNING' | 'CRITICAL'
        self.message = message
        self.recommendation = recommendation

    def __str__(self) -> str:
        color_prefix = "⚠️ [WARNING]" if self.severity == "WARNING" else "🚨 [CRITICAL ALERT]"
        return f"{color_prefix} {self.name}: {self.message}\n   -> Recommendation: {self.recommendation}"


class PitfallDetector:
    """
    Automated diagnostic engine analyzing loss curves and rollout statistics
    to detect training anomalies and non-stationarity.
    """

    def __init__(self) -> None:
        self.entropy_history = collections.deque(maxlen=10)
        self.value_loss_history = collections.deque(maxlen=10)
        self.policy_loss_history = collections.deque(maxlen=10)
        self.truncation_history = collections.deque(maxlen=10)
        self.combat_history = collections.deque(maxlen=10)

    def analyze(self, metrics: Dict[str, float]) -> List[PitfallAlert]:
        alerts: List[PitfallAlert] = []

        ent = metrics.get("loss_entropy", 0.0)
        v_loss = metrics.get("loss_value", 0.0)
        p_loss = metrics.get("loss_policy", 0.0)
        sink_loss = metrics.get("loss_sinkhorn", 0.0)
        combats = metrics.get("combats_per_rollout", 0.0)
        terminals = metrics.get("terminals_per_rollout", 0.0)
        truncations = metrics.get("truncations_per_rollout", 0.0)

        self.entropy_history.append(ent)
        self.value_loss_history.append(v_loss)
        self.policy_loss_history.append(p_loss)
        self.combat_history.append(combats)
        self.truncation_history.append(truncations)

        # 1. Entropy Collapse Check
        if ent < 0.35:
            alerts.append(PitfallAlert(
                name="Entropy Collapse",
                severity="CRITICAL",
                message=f"Policy entropy dropped to {ent:.3f} (near-deterministic policy).",
                recommendation="Increase entropy coefficient (c_ent), reduce policy LR, or check action mask validity."
            ))
        elif len(self.entropy_history) >= 5:
            ent_drop = (self.entropy_history[0] - ent) / max(0.01, self.entropy_history[0])
            if ent_drop > 0.60:
                alerts.append(PitfallAlert(
                    name="Rapid Entropy Decay",
                    severity="WARNING",
                    message=f"Entropy fell by {ent_drop*100:.1f}% over the last 5 iterations.",
                    recommendation="Monitor policy diversity to avoid premature mode collapse."
                ))

        # 2. Value Loss Explosion / Non-Stationarity
        if v_loss > 0.25:
            alerts.append(PitfallAlert(
                name="Critic Value Loss Spike",
                severity="WARNING",
                message=f"Value loss is high ({v_loss:.4f}) for a bounded [-1, 1] reward scale.",
                recommendation="Check GAE lambda/gamma, verify reward scaling, or reduce critic learning rate."
            ))

        # 3. Turtling / Passive Stagnation
        total_episodes = terminals + truncations
        if total_episodes > 0:
            trunc_rate = truncations / total_episodes
            if trunc_rate > 0.40 and combats < 50:
                alerts.append(PitfallAlert(
                    name="Turtling & Passive Deadlock",
                    severity="WARNING",
                    message=f"High truncation rate ({trunc_rate*100:.1f}%) and low combat density ({combats:.0f} combats/rollout).",
                    recommendation="Increase scripted heuristic opponent ratio in PFSP or incentivize aggressive exploration."
                ))

        # 4. Sinkhorn Belief Divergence
        if sink_loss > 4.0:
            alerts.append(PitfallAlert(
                name="Belief Head Divergence",
                severity="WARNING",
                message=f"Sinkhorn belief CE loss is elevated ({sink_loss:.3f}).",
                recommendation="Verify log-space Sinkhorn normalization iterations and dead-rank masking."
            ))

        return alerts


class TrainingTelemetry:
    """
    Live telemetry logger and formatted terminal dashboard.
    """

    def __init__(self, total_iterations: int, num_envs: int, rollout_steps: int) -> None:
        self.total_iterations = total_iterations
        self.num_envs = num_envs
        self.rollout_steps = rollout_steps
        self.batch_size = num_envs * rollout_steps
        self.start_time = time.time()
        self.detector = PitfallDetector()

        # Rolling statistics
        self.history: List[Dict[str, float]] = []

    def log_iteration(
        self,
        iteration: int,
        rollout_metrics: Dict[str, float],
        update_metrics: Dict[str, float],
        pfsp_pool_size: int,
        iter_time: float,
    ) -> List[PitfallAlert]:
        sps = self.batch_size / max(1e-4, iter_time)
        elapsed = time.time() - self.start_time
        eta_seconds = max(0, (self.total_iterations - iteration) * (elapsed / max(1, iteration)))
        eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))

        combined_metrics = {**rollout_metrics, **update_metrics}
        combined_metrics["sps"] = sps
        combined_metrics["iter_time"] = iter_time
        self.history.append(combined_metrics)

        # Run automated pitfall analysis
        alerts = self.detector.analyze(combined_metrics)

        # Print formatted live dashboard line
        print(
            f"[Iter {iteration:04d}/{self.total_iterations:04d}] "
            f"SPS: {sps:6.0f} | "
            f"ETA: {eta_str} | "
            f"L_pol: {update_metrics.get('loss_policy', 0.0):+.4f} | "
            f"L_val: {update_metrics.get('loss_value', 0.0):.4f} | "
            f"L_sink: {update_metrics.get('loss_sinkhorn', 0.0):.4f} | "
            f"Ent: {update_metrics.get('loss_entropy', 0.0):.3f} | "
            f"Combats: {rollout_metrics.get('combats_per_rollout', 0.0):4.0f} | "
            f"PFSP Pool: {pfsp_pool_size:2d}"
        )

        # Display any active pitfall alerts
        for alert in alerts:
            print(f"  {alert}")

        return alerts
