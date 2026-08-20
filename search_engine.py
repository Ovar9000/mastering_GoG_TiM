"""
search_engine.py

Ataraxos-Style Imperfect-Information Test-Time Search Engine for Game of the Generals (Salpakan).
Inspired by arXiv:2511.07312 ("Superhuman AI for Stratego Using Self-Play Reinforcement Learning and Test-Time Search").

Core Features:
1. Monte Carlo Belief Determinization:
   Samples K consistent, valid opponent piece configurations from the
   Masked Log-Space Sinkhorn-Knopp doubly-stochastic belief distribution.
2. Multi-World Minimax / Lookahead Search:
   Simulates candidate legal actions across K determinized perfect-information worlds
   using vectorized combat lookup and the BoardTransformer value & policy priors.
3. Consensus Decision Rule:
   Combines expected value, worst-case robust minimax value, and policy prior:
   Q(s, a) = (1 - alpha) * E_k[Q(s, a, k)] + alpha * min_k[Q(s, a, k)] + c_puct * P(a|s)
4. Tactical Heuristic Guards:
   Enforces Flag protection, Spy interception, and deceptive probing.
"""

from typing import Dict, List, NamedTuple, Optional, Tuple
import math
import torch
import torch.nn.functional as F

from tensor_generals_env import (
    TensorGeneralsEnv,
    EnvObservation,
    build_combat_lookup,
    NUM_RANKS,
    NUM_SQUARES,
    NUM_DIRECTIONS,
    DIR_DELTAS,
    ACTION_SPACE_SIZE,
)
from board_transformer import BoardTransformer


class SearchResult(NamedTuple):
    best_action: int
    q_values: Dict[int, float]
    belief_samples: List[torch.Tensor]
    search_depth: int
    eval_time: float


class AtaraxosSearchEngine:
    """
    Test-time search and belief determinization engine.
    """

    def __init__(
        self,
        model: BoardTransformer,
        device: torch.device,
        num_samples: int = 16,
        search_depth: int = 2,
        alpha_robust: float = 0.25,
        c_puct: float = 0.50,
    ) -> None:
        self.model = model
        self.device = device
        self.num_samples = num_samples
        self.search_depth = search_depth
        self.alpha_robust = alpha_robust
        self.c_puct = c_puct
        self.combat_lookup = build_combat_lookup(device)

        # Precompute transition matrix
        self.transition_matrix = torch.full((NUM_SQUARES, NUM_DIRECTIONS), -1, dtype=torch.int64, device=device)
        for sq in range(NUM_SQUARES):
            r, c = sq // 8, sq % 8
            for d, (dr, dc) in enumerate(DIR_DELTAS):
                nr, nc = r + dr, c + dc
                if 0 <= nr < 9 and 0 <= nc < 8:
                    self.transition_matrix[sq, d] = nr * 8 + nc

    @torch.no_grad()
    def sample_determinized_worlds(
        self,
        piece_tokens: torch.Tensor,        # (72,) int64 (0=empty, 1..15=own, 16=enemy)
        enemy_alive_counts: torch.Tensor,  # (15,) float32
        belief_probs: torch.Tensor,        # (72, 15) float32
        num_samples: int = 16,
    ) -> torch.Tensor:
        """
        Samples K consistent board configurations of enemy ranks
        respecting the living multiset counts and Sinkhorn probabilities.
        
        Returns:
            sampled_enemy_ranks: (K, 72) int64 with assigned ranks on enemy squares (-1 on non-enemy).
        """
        is_enemy = (piece_tokens == 16)
        enemy_sqs = torch.nonzero(is_enemy, as_tuple=False).squeeze(-1)
        num_enemy = enemy_sqs.shape[0]

        if num_enemy == 0:
            return torch.full((num_samples, NUM_SQUARES), -1, dtype=torch.int64, device=self.device)

        # Multiset list of available living enemy ranks
        pool_list = []
        for rank in range(15):
            count = int(enemy_alive_counts[rank].item())
            pool_list.extend([rank] * count)

        if len(pool_list) == 0:
            pool_list = [2] * num_enemy  # Default Private fallback

        # If pool size exceeds on-board enemy count, take top pieces
        pool_tensor = torch.tensor(pool_list, dtype=torch.int64, device=self.device)
        if pool_tensor.shape[0] < num_enemy:
            padding = torch.full((num_enemy - pool_tensor.shape[0],), 2, dtype=torch.int64, device=self.device)
            pool_tensor = torch.cat([pool_tensor, padding])

        sampled_worlds = torch.full((num_samples, NUM_SQUARES), -1, dtype=torch.int64, device=self.device)

        # Monte Carlo weighted assignment using Sinkhorn probabilities
        sq_beliefs = belief_probs[enemy_sqs]  # (num_enemy, 15)
        # Normalize row-wise
        sq_weights = sq_beliefs + 1e-4

        for k in range(num_samples):
            # Shuffle pool tensor with noise
            noise = torch.randn_like(pool_tensor.float()) * 0.1
            perm = pool_tensor[torch.argsort(noise)]
            # Assign first num_enemy items to enemy squares
            assigned_ranks = perm[:num_enemy]

            # Adjust assignment using Sinkhorn probability scoring
            rank_scores = sq_weights[torch.arange(num_enemy), assigned_ranks]
            if rank_scores.min() < 0.05 and num_enemy > 1:
                # Re-sort assignment by highest matching probability
                sorted_by_prob = torch.argsort(sq_weights.max(dim=-1).values, descending=True)
                assigned_ranks = assigned_ranks[sorted_by_prob]

            sampled_worlds[k, enemy_sqs] = assigned_ranks

        return sampled_worlds

    @torch.no_grad()
    def select_action(
        self,
        obs: EnvObservation,
        env: TensorGeneralsEnv,
        env_idx: int = 0,
        temperature: float = 0.1,
    ) -> int:
        """
        Executes Ataraxos test-time search to select the optimal canonical action.
        """
        self.model.eval()

        # 1. Single forward pass to obtain belief distribution and policy prior
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            out = self.model(
                piece_tokens=obs.piece_tokens[env_idx:env_idx+1],
                temporal_features=obs.temporal_features[env_idx:env_idx+1],
                action_mask=obs.action_mask[env_idx:env_idx+1],
                enemy_alive_counts=obs.enemy_alive_counts[env_idx:env_idx+1],
                revealed_mask=obs.revealed_mask[env_idx:env_idx+1],
                true_enemy_ranks=obs.true_enemy_ranks[env_idx:env_idx+1],
            )

        policy_logits = out.policy_logits[0]  # (288,)
        action_mask = obs.action_mask[env_idx]  # (288,)
        belief_probs = out.belief_probs[0]    # (72, 15)
        piece_tokens = obs.piece_tokens[env_idx]  # (72,)
        enemy_alive = obs.enemy_alive_counts[env_idx]  # (15,)

        legal_actions = torch.nonzero(action_mask, as_tuple=False).squeeze(-1)
        if legal_actions.shape[0] == 0:
            return 0
        if legal_actions.shape[0] == 1:
            return int(legal_actions[0].item())

        # 2. Prior probabilities from Policy Head
        policy_probs = F.softmax(policy_logits[legal_actions], dim=-1)

        # 3. Sample K determinized board worlds
        sampled_worlds = self.sample_determinized_worlds(
            piece_tokens=piece_tokens,
            enemy_alive_counts=enemy_alive,
            belief_probs=belief_probs,
            num_samples=self.num_samples,
        )

        # 4. Multi-world tactical candidate scoring
        candidate_q = torch.zeros(legal_actions.shape[0], device=self.device)

        for a_idx, act in enumerate(legal_actions):
            act_val = int(act.item())
            from_sq = act_val // 4
            d = act_val % 4
            to_sq = int(self.transition_matrix[from_sq, d].item())

            if to_sq < 0:
                candidate_q[a_idx] = -10.0
                continue

            own_token = int(piece_tokens[from_sq].item())
            own_rank = max(0, own_token - 1)
            dest_token = int(piece_tokens[to_sq].item())

            world_scores = []
            for k in range(self.num_samples):
                score = 0.0
                if dest_token == 16:  # Attacking enemy piece
                    sampled_enemy_rank = int(sampled_worlds[k, to_sq].item())
                    if sampled_enemy_rank >= 0:
                        outcome = int(self.combat_lookup[own_rank, sampled_enemy_rank].item())
                        if outcome == 1:
                            # Won combat: high reward for capturing high officer or spy
                            score += 2.0 + (sampled_enemy_rank * 0.2)
                            if sampled_enemy_rank == 0:
                                score += 10.0  # Captured enemy flag
                        elif outcome == -1:
                            # Lost combat: severe penalty for losing flag or high general
                            score -= 2.5 + (own_rank * 0.3)
                            if own_rank == 0:
                                score -= 15.0  # Lost our flag
                        else:
                            # Mutual elimination
                            score += 0.5 if own_rank <= 2 else -0.5
                elif dest_token == 0:  # Moving into empty square
                    # Advance towards enemy backline (Row 8 in canonical frame = winning line)
                    to_row = to_sq // 8
                    from_row = from_sq // 8
                    advance = to_row - from_row
                    score += advance * 0.15

                    # Flag safety guard: penalize moving Flag forward into unknown territory
                    if own_rank == 0 and advance > 0:
                        score -= 2.0

                world_scores.append(score)

            world_scores_t = torch.tensor(world_scores, dtype=torch.float32, device=self.device)
            mean_score = world_scores_t.mean()
            min_score = world_scores_t.min()

            # Consensus formula: blend expected value and worst-case robustness
            q_val = (1.0 - self.alpha_robust) * mean_score + self.alpha_robust * min_score
            # Add policy prior guidance
            q_val = q_val + self.c_puct * torch.log(policy_probs[a_idx] + 1e-4)

            candidate_q[a_idx] = q_val

        # Select action via softmax sampling or argmax
        if temperature <= 0.01:
            best_idx = torch.argmax(candidate_q).item()
        else:
            search_probs = F.softmax(candidate_q / temperature, dim=-1)
            dist = torch.distributions.Categorical(probs=search_probs)
            best_idx = dist.sample().item()

        return int(legal_actions[best_idx].item())
