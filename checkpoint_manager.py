"""
checkpoint_manager.py

Active Rolling Buffer Savepoints & Checkpoint Manager:
- Atomic saving and resumption of complete training state:
  * Model state_dict
  * Optimizer state_dict & LR Scheduler state
  * AMP GradScaler state
  * PFSP historical opponent pool and winrates
  * Training step, iteration, and rolling metrics
- Keeps rolling N latest checkpoints + top K best checkpoints (evaluated on winrate/rewards).
- Robust state restoration for seamless crash-recovery and long-running training continuation.
"""

from typing import Any, Dict, List, Optional, Tuple
import os
import json
import glob
import shutil
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from board_transformer import BoardTransformer


class CheckpointManager:
    """
    Manages active savepoints, rolling checkpoints, and top-K best model weights.
    """

    def __init__(
        self,
        checkpoint_dir: str = "checkpoints",
        max_rolling: int = 5,
        max_best: int = 3,
    ) -> None:
        self.checkpoint_dir = checkpoint_dir
        self.max_rolling = max_rolling
        self.max_best = max_best
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(os.path.join(self.checkpoint_dir, "best"), exist_ok=True)

        self.best_score: float = -float("inf")
        self.history_file = os.path.join(self.checkpoint_dir, "training_history.json")

    def save_checkpoint(
        self,
        iteration: int,
        model: BoardTransformer,
        optimizer: Optimizer,
        scheduler: Optional[_LRScheduler],
        scaler: Optional[torch.amp.GradScaler],
        pfsp_pool: Any,
        metrics: Dict[str, float],
        score: Optional[float] = None,
    ) -> str:
        """
        Saves a complete resumable state checkpoint.
        """
        checkpoint_data = {
            "iteration": iteration,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "scaler_state_dict": scaler.state_dict() if scaler else None,
            "pfsp_pool": [
                {
                    "step_idx": e.step_idx,
                    "wins": e.wins,
                    "games": e.games,
                    "state_dict": e.state_dict,
                }
                for e in pfsp_pool.pool
            ] if pfsp_pool else [],
            "metrics": metrics,
            "score": score,
        }

        # 1. Save latest rolling checkpoint
        ckpt_filename = f"checkpoint_iter_{iteration:06d}.pt"
        ckpt_path = os.path.join(self.checkpoint_dir, ckpt_filename)
        latest_path = os.path.join(self.checkpoint_dir, "latest.pt")

        torch.save(checkpoint_data, ckpt_path)
        # Update latest symlink / copy
        torch.save(checkpoint_data, latest_path)

        # 2. Check if this is the best checkpoint
        current_score = score if score is not None else -metrics.get("loss_policy", 0.0)
        is_best = current_score > self.best_score
        if is_best:
            self.best_score = current_score
            best_filename = f"best_model_iter_{iteration:06d}_score_{current_score:+.3f}.pt"
            best_path = os.path.join(self.checkpoint_dir, "best", best_filename)
            best_link = os.path.join(self.checkpoint_dir, "best_model.pt")
            torch.save(checkpoint_data, best_path)
            torch.save(checkpoint_data, best_link)
            self._prune_best_checkpoints()

        # 3. Prune old rolling checkpoints
        self._prune_rolling_checkpoints()

        # 4. Append to training history JSON
        self._update_history(iteration, metrics, current_score)

        return ckpt_path

    def load_checkpoint(
        self,
        checkpoint_path: Optional[str] = None,
        model: Optional[BoardTransformer] = None,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[_LRScheduler] = None,
        scaler: Optional[torch.amp.GradScaler] = None,
        pfsp_pool: Optional[Any] = None,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Restores full training state from a specified checkpoint path or latest.pt.
        """
        if checkpoint_path is None:
            checkpoint_path = os.path.join(self.checkpoint_dir, "latest.pt")

        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        print(f"[CheckpointManager] Loading state from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        iteration = checkpoint.get("iteration", 0)

        # Restore Model
        if model is not None and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            print(f"[CheckpointManager] Restored model weights (Iter {iteration})")

        # Restore Optimizer
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            print(f"[CheckpointManager] Restored optimizer state")

        # Restore Scheduler
        if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            print(f"[CheckpointManager] Restored LR scheduler state")

        # Restore Scaler
        if scaler is not None and checkpoint.get("scaler_state_dict") is not None:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
            print(f"[CheckpointManager] Restored GradScaler state")

        # Restore PFSP Opponent Pool
        if pfsp_pool is not None and "pfsp_pool" in checkpoint:
            from train_pfsp import CheckpointEntry
            pfsp_pool.pool.clear()
            for entry_data in checkpoint["pfsp_pool"]:
                entry = CheckpointEntry(entry_data["state_dict"], entry_data["step_idx"])
                entry.wins = entry_data["wins"]
                entry.games = entry_data["games"]
                pfsp_pool.pool.append(entry)
            print(f"[CheckpointManager] Restored PFSP opponent pool ({len(pfsp_pool.pool)} models)")

        return iteration, checkpoint.get("metrics", {})

    def _prune_rolling_checkpoints(self) -> None:
        """Keeps only the most recent max_rolling checkpoint files."""
        pattern = os.path.join(self.checkpoint_dir, "checkpoint_iter_*.pt")
        files = sorted(glob.glob(pattern))
        if len(files) > self.max_rolling:
            for old_file in files[:-self.max_rolling]:
                try:
                    os.remove(old_file)
                except OSError:
                    pass

    def _prune_best_checkpoints(self) -> None:
        """Keeps only the top max_best best checkpoint files."""
        pattern = os.path.join(self.checkpoint_dir, "best", "best_model_iter_*.pt")
        files = sorted(glob.glob(pattern))
        if len(files) > self.max_best:
            for old_file in files[:-self.max_best]:
                try:
                    os.remove(old_file)
                except OSError:
                    pass

    def _update_history(self, iteration: int, metrics: Dict[str, float], score: float) -> None:
        """Appends metrics record to history log."""
        record = {
            "iteration": iteration,
            "score": score,
            "metrics": metrics,
        }
        history = []
        if os.path.isfile(self.history_file):
            try:
                with open(self.history_file, "r") as f:
                    history = json.load(f)
            except Exception:
                history = []
        history.append(record)
        with open(self.history_file, "w") as f:
            json.dump(history, f, indent=2)
