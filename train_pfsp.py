"""
train_pfsp.py

Production-Grade GAE-PPO Training Pipeline with:
- Prioritized Fictitious Self-Play (PFSP) with up to 50 historical checkpoints.
- Priority sampling: w_i = (1 - win_rate_i)^1.5 + eps.
- Disjoint Environment Partitioning (30% Live Self-Play, 50% PFSP Pool, 20% Vectorized Heuristic).
- Vectorized GPU Heuristic Opponent (zero CPU overhead).
- Native AMP bfloat16 mixed precision training on CUDA.
- GAE (gamma=0.99, lambda=0.95), PPO clip=0.2, Value clip=0.2.
- Total loss = L_policy + 0.5 * L_value + 0.25 * L_sinkhorn_ce - 0.01 * L_entropy.
"""

from typing import Dict, List, NamedTuple, Optional, Tuple
import os
import copy
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from tensor_generals_env import (
    TensorGeneralsEnv,
    EnvObservation,
    NUM_SQUARES,
    NUM_DIRECTIONS,
    ACTION_SPACE_SIZE,
)
from board_transformer import BoardTransformer, ModelOutput


class VectorizedHeuristicAgent:
    """
    Vectorized rule-based / greedy-heuristic agent running 100% on CUDA.
    Evaluates legal actions via tensorized scoring without per-env Python loops.
    """

    def __init__(self, device: torch.device) -> None:
        self.device = device
        # Direction bias: North (0) moves towards enemy backline in canonical perspective (+1.5 score)
        self.dir_scores = torch.tensor([1.5, 0.0, 0.5, 0.5], dtype=torch.float32, device=device)

    @torch.no_grad()
    def select_action(
        self,
        obs: EnvObservation,
        env: TensorGeneralsEnv,
        env_indices: torch.Tensor,
        temperature: float = 0.5,
    ) -> torch.Tensor:
        """
        Selects actions for specified env_indices in canonical frame.
        """
        num_selected = env_indices.shape[0]
        if num_selected == 0:
            return torch.empty(0, dtype=torch.int64, device=self.device)

        # Action mask for selected envs: (N, 288)
        mask = obs.action_mask[env_indices]

        # Base scoring: Direction bias expanded to (N, 72, 4)
        scores = self.dir_scores.unsqueeze(0).unsqueeze(0).expand(num_selected, NUM_SQUARES, NUM_DIRECTIONS).clone()

        # Piece tokens on canonical board: (N, 72)
        tokens = obs.piece_tokens[env_indices]  # 0=empty, 1..15=own, 16=enemy

        # Destination squares in canonical frame
        from_sqs = torch.arange(NUM_SQUARES, device=self.device).unsqueeze(0).unsqueeze(2)  # (1, 72, 1)
        dirs = torch.arange(NUM_DIRECTIONS, device=self.device).unsqueeze(0).unsqueeze(1)    # (1, 1, 4)
        to_sqs = env.transition_matrix[from_sqs, dirs]  # (1, 72, 4)

        safe_to = torch.clamp(to_sqs, min=0)
        dest_tokens = torch.gather(tokens.unsqueeze(1).expand(-1, NUM_SQUARES, -1), 2, safe_to)  # (N, 72, 4)

        # 1. Reward attacking enemy pieces (+4.0)
        is_attack = (dest_tokens == 16)
        scores = scores + torch.where(is_attack, torch.full_like(scores, 4.0), torch.zeros_like(scores))

        # 2. Reward aggressive attacks with Spies (token 2) or High Officers (tokens 11..15) (+3.0)
        moving_ranks = tokens.unsqueeze(2).expand(-1, -1, NUM_DIRECTIONS)  # (N, 72, 4)
        is_aggressive_piece = (moving_ranks == 2) | (moving_ranks >= 11)
        scores = scores + torch.where(is_attack & is_aggressive_piece, torch.full_like(scores, 3.0), torch.zeros_like(scores))

        # 3. Penalize Flag (token 1) moving into danger (-8.0)
        is_flag = (moving_ranks == 1)
        scores = scores + torch.where(is_flag & is_attack, torch.full_like(scores, -8.0), torch.zeros_like(scores))

        # Flatten scores to (N, 288) and apply legal action mask
        flat_scores = scores.view(num_selected, ACTION_SPACE_SIZE)
        masked_scores = torch.where(mask, flat_scores, torch.full_like(flat_scores, -1e4))

        # Sample with temperature-scaled categorical distribution
        probs = F.softmax(masked_scores / temperature, dim=-1)
        dist = torch.distributions.Categorical(probs=probs)
        actions = dist.sample()

        return actions


class CheckpointEntry:
    """Historical model entry in the PFSP pool."""
    def __init__(self, model_state_dict: Dict[str, torch.Tensor], step_idx: int) -> None:
        self.state_dict = copy.deepcopy(model_state_dict)
        self.step_idx = step_idx
        self.wins: float = 1.0  # Laplace smoothing
        self.games: float = 2.0

    @property
    def win_rate(self) -> float:
        return self.wins / max(1.0, self.games)

    @property
    def pfsp_weight(self) -> float:
        # Priority weight w_i = (1 - win_rate)^1.5 + 0.05
        return ((1.0 - self.win_rate) ** 1.5) + 0.05


class PFSPPool:
    """Manages the historical opponent pool for Prioritized Fictitious Self-Play."""

    def __init__(self, max_size: int = 50, device: torch.device = torch.device("cuda")) -> None:
        self.max_size = max_size
        self.device = device
        self.pool: List[CheckpointEntry] = []
        self.cached_models: Dict[int, BoardTransformer] = {}

    def add_checkpoint(self, model: BoardTransformer, step_idx: int) -> None:
        if len(self.pool) >= self.max_size:
            # Drop lowest priority or oldest checkpoint
            self.pool.pop(0)

        entry = CheckpointEntry(model.state_dict(), step_idx)
        self.pool.append(entry)

    def sample_opponent(self) -> Optional[CheckpointEntry]:
        if not self.pool:
            return None
        weights = torch.tensor([e.pfsp_weight for e in self.pool], dtype=torch.float32)
        idx = torch.multinomial(weights, num_samples=1).item()
        return self.pool[idx]

    def update_match_result(self, entry: CheckpointEntry, opponent_won: bool) -> None:
        entry.games += 1.0
        if opponent_won:
            entry.wins += 1.0


class RolloutBuffer:
    """
    On-device tensor buffer storing experience for PPO updates.
    """

    def __init__(self, num_steps: int, num_envs: int, device: torch.device) -> None:
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.device = device

        self.piece_tokens = torch.zeros((num_steps, num_envs, NUM_SQUARES), dtype=torch.int64, device=device)
        self.temporal_features = torch.zeros((num_steps, num_envs, NUM_SQUARES, 8), dtype=torch.float32, device=device)
        self.action_mask = torch.zeros((num_steps, num_envs, ACTION_SPACE_SIZE), dtype=torch.bool, device=device)
        self.enemy_alive = torch.zeros((num_steps, num_envs, 15), dtype=torch.float32, device=device)
        self.revealed_mask = torch.zeros((num_steps, num_envs, NUM_SQUARES), dtype=torch.bool, device=device)
        self.true_enemy_ranks = torch.zeros((num_steps, num_envs, NUM_SQUARES), dtype=torch.int64, device=device)

        self.actions = torch.zeros((num_steps, num_envs), dtype=torch.int64, device=device)
        self.log_probs = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
        self.values = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
        self.dones = torch.zeros((num_steps, num_envs), dtype=torch.bool, device=device)
        self.truncated = torch.zeros((num_steps, num_envs), dtype=torch.bool, device=device)

        self.advantages = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
        self.returns = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)

        self.step_idx = 0

    def insert(
        self,
        obs: EnvObservation,
        action: torch.Tensor,
        log_prob: torch.Tensor,
        reward: torch.Tensor,
        value: torch.Tensor,
        done: torch.Tensor,
        truncated: torch.Tensor,
    ) -> None:
        t = self.step_idx
        self.piece_tokens[t] = obs.piece_tokens
        self.temporal_features[t] = obs.temporal_features
        self.action_mask[t] = obs.action_mask
        self.enemy_alive[t] = obs.enemy_alive_counts
        self.revealed_mask[t] = obs.revealed_mask
        self.true_enemy_ranks[t] = obs.true_enemy_ranks

        self.actions[t] = action
        self.log_probs[t] = log_prob
        self.rewards[t] = reward
        self.values[t] = value.squeeze(-1)
        self.dones[t] = done
        self.truncated[t] = truncated

        self.step_idx += 1

    def compute_gae(self, last_value: torch.Tensor, gamma: float = 0.99, gae_lambda: float = 0.95) -> None:
        """
        Computes Generalized Advantage Estimation with critic bootstrap on truncation.
        """
        last_gae = torch.zeros(self.num_envs, device=self.device)
        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                next_non_terminal = (~self.dones[t]).float()
                next_val = last_value.squeeze(-1)
            else:
                next_non_terminal = (~self.dones[t]).float()
                next_val = self.values[t + 1]

            # If truncated, bootstrap value V(s_T)
            effective_reward = torch.where(
                self.truncated[t],
                self.rewards[t] + gamma * next_val,
                self.rewards[t]
            )

            delta = effective_reward + gamma * next_val * next_non_terminal - self.values[t]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            self.advantages[t] = last_gae

        self.returns = self.advantages + self.values


class PFSPTrainer:
    """
    Main PPO + PFSP Training Orchestrator.
    """

    def __init__(
        self,
        num_envs: int = 512,
        rollout_steps: int = 64,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_coef: float = 0.2,
        vf_clip_coef: float = 0.2,
        c_val: float = 0.5,
        c_sinkhorn: float = 0.25,
        c_ent: float = 0.01,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 4,
        minibatch_size: int = 2048,
        device: str = "cuda",
    ) -> None:
        self.device = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
        self.num_envs = num_envs
        self.rollout_steps = rollout_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.vf_clip_coef = vf_clip_coef
        self.c_val = c_val
        self.c_sinkhorn = c_sinkhorn
        self.c_ent = c_ent
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.minibatch_size = minibatch_size

        # 1. Vectorized Environment
        self.env = TensorGeneralsEnv(num_envs=num_envs, device=str(self.device))

        # 2. Main Policy Network
        self.model = BoardTransformer(
            d_model=192,
            nhead=6,
            num_layers=6,
            dim_feedforward=512,
        ).to(self.device)

        # 3. Opponents
        self.heuristic_agent = VectorizedHeuristicAgent(self.device)
        self.pfsp_pool = PFSPPool(max_size=50, device=self.device)
        self.opponent_model = copy.deepcopy(self.model).to(self.device)
        self.opponent_model.eval()

        # 4. Partition Environment Buckets
        # Bucket A: 30% Pure Self-Play (Live Model)
        # Bucket B: 50% Historical PFSP Pool
        # Bucket C: 20% Vectorized Heuristic
        n_a = int(0.30 * num_envs)
        n_b = int(0.50 * num_envs)
        n_c = num_envs - n_a - n_b

        self.bucket_a_idx = torch.arange(0, n_a, device=self.device)
        self.bucket_b_idx = torch.arange(n_a, n_a + n_b, device=self.device)
        self.bucket_c_idx = torch.arange(n_a + n_b, num_envs, device=self.device)

        # 5. Optimizer & Scheduler
        self.optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.buffer = RolloutBuffer(rollout_steps, num_envs, self.device)

        # 6. Mixed Precision Scaler
        self.scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    def collect_rollout(self, obs: EnvObservation) -> Tuple[EnvObservation, Dict[str, float]]:
        """
        Collects T rollout steps concurrently across all B environments.
        """
        self.model.eval()
        self.buffer.step_idx = 0

        total_combats = 0.0
        total_terminals = 0.0

        for step in range(self.rollout_steps):
            # Opponent Action Selection for Player 2 when playing vs baseline/historical
            is_p1 = (obs.current_player == 0)
            is_p2 = (obs.current_player == 1)

            actions = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
            log_probs = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
            values = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

            # Player 1 (Main Learner) in all envs & Player 2 in Bucket A (Pure Self-Play)
            main_envs = torch.nonzero(is_p1 | (is_p2 & torch.isin(torch.arange(self.num_envs, device=self.device), self.bucket_a_idx)), as_tuple=False).squeeze(-1)

            if main_envs.shape[0] > 0:
                with torch.no_grad():
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                        act, lp, _, val, _ = self.model.get_action_and_value(
                            piece_tokens=obs.piece_tokens[main_envs],
                            temporal_features=obs.temporal_features[main_envs],
                            action_mask=obs.action_mask[main_envs],
                            enemy_alive_counts=obs.enemy_alive_counts[main_envs],
                            revealed_mask=obs.revealed_mask[main_envs],
                            true_enemy_ranks=obs.true_enemy_ranks[main_envs],
                        )
                actions[main_envs] = act
                log_probs[main_envs] = lp.float()
                values[main_envs] = val.squeeze(-1).float()

            # Player 2 in Bucket B (Historical PFSP Opponents)
            pfsp_p2 = torch.nonzero(is_p2 & torch.isin(torch.arange(self.num_envs, device=self.device), self.bucket_b_idx), as_tuple=False).squeeze(-1)
            if pfsp_p2.shape[0] > 0:
                with torch.no_grad():
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                        act, lp, _, val, _ = self.opponent_model.get_action_and_value(
                            piece_tokens=obs.piece_tokens[pfsp_p2],
                            temporal_features=obs.temporal_features[pfsp_p2],
                            action_mask=obs.action_mask[pfsp_p2],
                            enemy_alive_counts=obs.enemy_alive_counts[pfsp_p2],
                            revealed_mask=obs.revealed_mask[pfsp_p2],
                        )
                actions[pfsp_p2] = act

            # Player 2 in Bucket C (Vectorized Heuristic Opponent)
            heur_p2 = torch.nonzero(is_p2 & torch.isin(torch.arange(self.num_envs, device=self.device), self.bucket_c_idx), as_tuple=False).squeeze(-1)
            if heur_p2.shape[0] > 0:
                actions[heur_p2] = self.heuristic_agent.select_action(obs, self.env, heur_p2)

            # Step environment
            next_obs, rewards, terminated, truncated, info = self.env.step(actions)
            done = terminated | truncated

            total_combats += info["is_combat"].sum().item()
            total_terminals += terminated.sum().item()

            # Record in rollout buffer
            self.buffer.insert(
                obs=obs,
                action=actions,
                log_prob=log_probs,
                reward=rewards,
                value=values.unsqueeze(-1),
                done=done,
                truncated=truncated,
            )

            obs = next_obs

        # Bootstrap final value for GAE
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                output = self.model(
                    piece_tokens=obs.piece_tokens,
                    temporal_features=obs.temporal_features,
                    action_mask=obs.action_mask,
                    enemy_alive_counts=obs.enemy_alive_counts,
                    revealed_mask=obs.revealed_mask,
                )
                last_value = output.value.float()

        self.buffer.compute_gae(last_value, self.gamma, self.gae_lambda)

        metrics = {
            "combats_per_rollout": total_combats,
            "terminals_per_rollout": total_terminals,
        }
        return obs, metrics

    def update_ppo(self) -> Dict[str, float]:
        """
        Performs mini-batch PPO optimization step with AMP bfloat16.
        """
        self.model.train()

        # Flatten batch
        b_tokens = self.buffer.piece_tokens.view(-1, NUM_SQUARES)
        b_features = self.buffer.temporal_features.view(-1, NUM_SQUARES, 8)
        b_masks = self.buffer.action_mask.view(-1, ACTION_SPACE_SIZE)
        b_enemy_alive = self.buffer.enemy_alive.view(-1, 15)
        b_revealed = self.buffer.revealed_mask.view(-1, NUM_SQUARES)
        b_true_ranks = self.buffer.true_enemy_ranks.view(-1, NUM_SQUARES)

        b_actions = self.buffer.actions.view(-1)
        b_old_log_probs = self.buffer.log_probs.view(-1)
        b_advantages = self.buffer.advantages.view(-1)
        b_returns = self.buffer.returns.view(-1)
        b_old_values = self.buffer.values.view(-1)

        # Normalize advantages
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        total_samples = b_actions.shape[0]
        batch_indices = torch.arange(total_samples, device=self.device)

        policy_losses = []
        value_losses = []
        sinkhorn_losses = []
        entropy_losses = []

        for epoch in range(self.ppo_epochs):
            perm = torch.randperm(total_samples, device=self.device)
            for start in range(0, total_samples, self.minibatch_size):
                end = min(start + self.minibatch_size, total_samples)
                mb_idx = perm[start:end]

                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                    _, new_log_prob, entropy, new_value, sinkhorn_loss = self.model.get_action_and_value(
                        piece_tokens=b_tokens[mb_idx],
                        temporal_features=b_features[mb_idx],
                        action_mask=b_masks[mb_idx],
                        enemy_alive_counts=b_enemy_alive[mb_idx],
                        revealed_mask=b_revealed[mb_idx],
                        action=b_actions[mb_idx],
                        true_enemy_ranks=b_true_ranks[mb_idx],
                    )

                    new_val_flat = new_value.squeeze(-1)

                    # Policy Loss (PPO Clipped Objective)
                    log_ratio = new_log_prob - b_old_log_probs[mb_idx]
                    ratio = torch.exp(log_ratio)
                    mb_adv = b_advantages[mb_idx]

                    surr1 = ratio * mb_adv
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_coef, 1.0 + self.clip_coef) * mb_adv
                    loss_policy = -torch.min(surr1, surr2).mean()

                    # Value Loss (Clipped Value Objective)
                    mb_returns = b_returns[mb_idx]
                    val_clipped = b_old_values[mb_idx] + torch.clamp(
                        new_val_flat - b_old_values[mb_idx],
                        -self.vf_clip_coef,
                        self.vf_clip_coef
                    )
                    v_loss_unclipped = (new_val_flat - mb_returns) ** 2
                    v_loss_clipped = (val_clipped - mb_returns) ** 2
                    loss_value = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                    # Entropy & Belief Losses
                    loss_entropy = entropy.mean()
                    loss_total = (
                        loss_policy
                        + self.c_val * loss_value
                        + self.c_sinkhorn * sinkhorn_loss
                        - self.c_ent * loss_entropy
                    )

                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(loss_total).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()

                policy_losses.append(loss_policy.item())
                value_losses.append(loss_value.item())
                sinkhorn_losses.append(sinkhorn_loss.item())
                entropy_losses.append(loss_entropy.item())

        return {
            "loss_policy": float(torch.tensor(policy_losses).mean()),
            "loss_value": float(torch.tensor(value_losses).mean()),
            "loss_sinkhorn": float(torch.tensor(sinkhorn_losses).mean()),
            "loss_entropy": float(torch.tensor(entropy_losses).mean()),
        }

    def train(self, total_iterations: int = 100, checkpoint_interval: int = 10) -> None:
        """
        Executes end-to-end self-play training loop with PFSP checkpointing.
        """
        print(f"=== Starting PFSP Training on {self.device} ===")
        print(f"Envs: {self.num_envs} | Rollout: {self.rollout_steps} | Batch: {self.num_envs * self.rollout_steps} steps/iter")

        obs = self.env.get_canonical_observation()
        start_time = time.time()

        for iteration in range(1, total_iterations + 1):
            t0 = time.time()
            obs, rollout_metrics = self.collect_rollout(obs)
            update_metrics = self.update_ppo()
            iter_time = time.time() - t0

            steps_per_sec = (self.num_envs * self.rollout_steps) / iter_time

            # Update PFSP Pool
            if iteration % checkpoint_interval == 0:
                self.pfsp_pool.add_checkpoint(self.model, iteration)
                sampled_entry = self.pfsp_pool.sample_opponent()
                if sampled_entry is not None:
                    self.opponent_model.load_state_dict(sampled_entry.state_dict)

            print(
                f"[Iter {iteration:03d}/{total_iterations}] "
                f"SPS: {steps_per_sec:6.0f} | "
                f"L_pol: {update_metrics['loss_policy']:+.4f} | "
                f"L_val: {update_metrics['loss_value']:.4f} | "
                f"L_sink: {update_metrics['loss_sinkhorn']:.4f} | "
                f"Ent: {update_metrics['loss_entropy']:.3f} | "
                f"Combats: {rollout_metrics['combats_per_rollout']:3.0f} | "
                f"PFSP Pool: {len(self.pfsp_pool.pool)}"
            )

        total_elapsed = time.time() - start_time
        print(f"=== Training Completed in {total_elapsed:.2f}s ===")


if __name__ == "__main__":
    trainer = PFSPTrainer(num_envs=512, rollout_steps=64)
    trainer.train(total_iterations=5, checkpoint_interval=2)
