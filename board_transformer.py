"""
board_transformer.py

73-Token Pre-LN Transformer Architecture with:
- 1 Learnable [CLS] Token + 72 Square Tokens.
- Token Representation = Piece_Embedding(17) + Spatial_Pos(72) + Linear(Temporal_Dim(8) -> d_model).
- 6-Layer Pre-LN Transformer Encoder using F.scaled_dot_product_attention.
- Heads:
  1. Policy Head: Linear(192 -> 4) per square -> Flatten to (B, 288) with legal action masking.
  2. Value Head: [CLS] -> MLP(192 -> 64 -> 1) with Tanh producing scalar V(s) in [-1, 1].
  3. Masked Sinkhorn-Knopp Belief Head: Linear(192 -> 15) with exact dead-rank masking,
     revealed-piece deduction, and log-space doubly-stochastic optimal transport.
"""

from typing import Dict, NamedTuple, Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ModelOutput(NamedTuple):
    policy_logits: torch.Tensor       # (B, 288) masked logits
    value: torch.Tensor               # (B, 1) scalar in [-1, 1]
    belief_probs: torch.Tensor        # (B, 72, 15) Sinkhorn doubly-stochastic belief distribution
    belief_log_probs: torch.Tensor    # (B, 72, 15) log-space beliefs
    sinkhorn_loss: torch.Tensor       # scalar cross-entropy loss


class TransformerEncoderLayerPreLN(nn.Module):
    """
    Standard Pre-LN Transformer Encoder layer with scaled_dot_product_attention.
    """

    def __init__(self, d_model: int = 192, nhead: int = 6, dim_feedforward: int = 512, dropout: float = 0.0) -> None:
        super().__init__()
        assert d_model % nhead == 0, f"d_model {d_model} must be divisible by nhead {nhead}"
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead

        self.norm1 = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.norm2 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LN Self-Attention
        norm_x = self.norm1(x)
        B, S, D = norm_x.shape

        q = self.q_proj(norm_x).view(B, S, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_proj(norm_x).view(B, S, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_proj(norm_x).view(B, S, self.nhead, self.head_dim).transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, D)
        x = x + self.dropout(self.out_proj(attn_out))

        # Pre-LN MLP
        norm_x2 = self.norm2(x)
        mlp_out = self.linear2(self.dropout(self.act(self.linear1(norm_x2))))
        x = x + self.dropout(mlp_out)

        return x


class BoardTransformer(nn.Module):
    """
    73-Token Transformer Network for Game of the Generals (Salpakan).
    """

    def __init__(
        self,
        d_model: int = 192,
        nhead: int = 6,
        num_layers: int = 6,
        dim_feedforward: int = 512,
        num_ranks: int = 15,
        num_squares: int = 72,
        num_piece_tokens: int = 17,  # 0=Empty, 1..15=Own pieces, 16=Enemy piece
        temporal_dim: int = 8,
        sinkhorn_iters: int = 10,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_squares = num_squares
        self.num_ranks = num_ranks
        self.sinkhorn_iters = sinkhorn_iters

        # 1. Embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.piece_emb = nn.Embedding(num_piece_tokens, d_model)
        self.spatial_pos_emb = nn.Embedding(num_squares, d_model)
        self.temporal_proj = nn.Linear(temporal_dim, d_model)

        # 2. Transformer Encoder (6 Pre-LN layers)
        self.layers = nn.ModuleList([
            TransformerEncoderLayerPreLN(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
            )
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

        # 3. Belief Prior Head: Linear(192 -> 15) to produce cost matrix for Sinkhorn-Knopp
        self.belief_head = nn.Linear(d_model, num_ranks)
        self.belief_proj = nn.Linear(num_ranks, d_model)

        # 4. Policy Head: Belief-Conditioned MLP(192 * 2 -> 192 -> 4) per square token
        self.policy_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 4)
        )

        # 5. Value Head: Belief-Conditioned MLP(192 * 2 -> 64 -> 1) -> Tanh()
        self.value_head = nn.Sequential(
            nn.Linear(d_model * 2, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )

        # 6. Army Deployment Scoring Head: Linear(192 -> 1) per home square
        self.deployment_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

        self.register_buffer("square_indices", torch.arange(num_squares, dtype=torch.int64))

    def _masked_sinkhorn_knopp(
        self,
        raw_logits: torch.Tensor,        # (B, 72, 15)
        piece_tokens: torch.Tensor,      # (B, 72)
        enemy_alive_counts: torch.Tensor,# (B, 15)
        revealed_mask: torch.Tensor,     # (B, 72)
        true_enemy_ranks: Optional[torch.Tensor] = None, # (B, 72)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Applies log-space Masked Sinkhorn-Knopp doubly-stochastic normalization.
        
        Guarantees:
        1. Friendly and empty squares are masked out to 0 probability.
        2. Dead ranks (count = 0) are masked out with zero probability.
        3. Revealed pieces are deducted from target column marginals.
        4. Unknown enemy square row sum = 1.0, alive rank column sum = target count.
        """
        B, S, R = raw_logits.shape
        device = raw_logits.device

        # Mask of unknown enemy pieces on board
        is_enemy = (piece_tokens == 16)
        is_unknown_enemy = is_enemy & (~revealed_mask)  # (B, 72)
        is_revealed_enemy = is_enemy & revealed_mask    # (B, 72)

        # Calculate revealed counts per rank to deduct from total living enemy counts
        target_counts = enemy_alive_counts.clone()  # (B, 15)
        if true_enemy_ranks is not None and is_revealed_enemy.any():
            safe_ranks = torch.clamp(true_enemy_ranks, min=0)
            rev_one_hot = F.one_hot(safe_ranks, num_classes=R).float() * is_revealed_enemy.unsqueeze(-1).float()
            revealed_counts_per_rank = rev_one_hot.sum(dim=1)  # (B, 15)
            target_counts = torch.clamp(enemy_alive_counts - revealed_counts_per_rank, min=0.0)

        # Mask of alive ranks for unrevealed matching
        alive_rank_mask = (target_counts > 0.0)  # (B, 15)

        # Construct log cost matrix M of shape (B, 72, 15)
        # Non-unknown squares and dead ranks receive large negative values (-1e4)
        M = raw_logits.clone()

        invalid_sq_mask = ~is_unknown_enemy.unsqueeze(-1).expand(B, S, R)
        invalid_rank_mask = ~alive_rank_mask.unsqueeze(1).expand(B, S, R)
        invalid_mask = invalid_sq_mask | invalid_rank_mask

        M = torch.where(invalid_mask, torch.full_like(M, -1e4), M)

        # Log target marginals
        log_col_target = torch.log(torch.clamp(target_counts, min=1e-6)).unsqueeze(1)  # (B, 1, 15)
        log_row_target = torch.zeros((B, S, 1), device=device)                         # (B, 72, 1)

        # Log-space Sinkhorn iterations
        for _ in range(self.sinkhorn_iters):
            # Row normalization: logsumexp over ranks
            log_row_sum = torch.logsumexp(M, dim=-1, keepdim=True)  # (B, 72, 1)
            row_diff = torch.where(is_unknown_enemy.unsqueeze(-1), log_row_sum - log_row_target, torch.zeros_like(log_row_sum))
            M = M - row_diff

            # Column normalization: logsumexp over squares
            log_col_sum = torch.logsumexp(M, dim=-2, keepdim=True)  # (B, 1, 15)
            col_diff = torch.where(alive_rank_mask.unsqueeze(1), log_col_sum - log_col_target, torch.zeros_like(log_col_sum))
            M = M - col_diff

        # Final row normalization so each unknown square's distribution strictly sums to 1.0
        final_log_row_sum = torch.logsumexp(M, dim=-1, keepdim=True)
        log_probs = M - final_log_row_sum
        probs = torch.exp(log_probs)

        # Apply strict zeroing to invalid entries
        probs = torch.where(invalid_mask, torch.zeros_like(probs), probs)
        log_probs = torch.where(invalid_mask, torch.full_like(log_probs, -1e4), log_probs)

        # For revealed enemy pieces, assign one-hot distribution if true rank provided
        if true_enemy_ranks is not None and is_revealed_enemy.any():
            rev_mask_3d = is_revealed_enemy.unsqueeze(-1).expand(B, S, R)
            safe_rev_ranks = torch.clamp(true_enemy_ranks, min=0)
            one_hot_rev = F.one_hot(safe_rev_ranks, num_classes=R).to(dtype=probs.dtype)

            probs = torch.where(rev_mask_3d, one_hot_rev, probs)
            log_probs = torch.where(rev_mask_3d, torch.full_like(log_probs, -1e4), log_probs)
            log_probs = torch.where(rev_mask_3d & (one_hot_rev > 0.5), torch.zeros_like(log_probs), log_probs)

        # Compute Sinkhorn CE Loss against privileged true enemy ranks
        if true_enemy_ranks is not None and is_unknown_enemy.any():
            valid_targets = true_enemy_ranks[is_unknown_enemy]  # (N,)
            valid_log_preds = log_probs[is_unknown_enemy]       # (N, 15)
            sinkhorn_loss = F.nll_loss(valid_log_preds, valid_targets)
        else:
            sinkhorn_loss = torch.tensor(0.0, device=device, dtype=raw_logits.dtype)

        return probs, log_probs, sinkhorn_loss

    def forward(
        self,
        piece_tokens: torch.Tensor,        # (B, 72) int64
        temporal_features: torch.Tensor,   # (B, 72, 8) float32
        action_mask: torch.Tensor,         # (B, 288) bool
        enemy_alive_counts: torch.Tensor,  # (B, 15) float32
        revealed_mask: torch.Tensor,       # (B, 72) bool
        true_enemy_ranks: Optional[torch.Tensor] = None, # (B, 72) int64
    ) -> ModelOutput:
        B = piece_tokens.shape[0]

        # 1. Square token embeddings
        piece_e = self.piece_emb(piece_tokens)                     # (B, 72, d_model)
        spatial_e = self.spatial_pos_emb(self.square_indices)      # (72, d_model)
        temporal_e = self.temporal_proj(temporal_features)         # (B, 72, d_model)

        square_tokens = piece_e + spatial_e.unsqueeze(0) + temporal_e  # (B, 72, d_model)

        # 2. Prepend [CLS] token -> (B, 73, d_model)
        cls_expanded = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls_expanded, square_tokens], dim=1)

        # 3. Transformer Backbone
        for layer in self.layers:
            tokens = layer(tokens)
        tokens = self.final_norm(tokens)

        # 4. Masked Sinkhorn-Knopp Optimal Transport Belief Head
        board_tokens = tokens[:, 1:]  # (B, 72, d_model)
        cls_out = tokens[:, 0]        # (B, d_model)

        belief_raw = self.belief_head(board_tokens)  # (B, 72, 15)
        belief_probs, belief_log_probs, sinkhorn_loss = self._masked_sinkhorn_knopp(
            raw_logits=belief_raw,
            piece_tokens=piece_tokens,
            enemy_alive_counts=enemy_alive_counts,
            revealed_mask=revealed_mask,
            true_enemy_ranks=true_enemy_ranks,
        )

        # 5. Project Doubly-Stochastic Beliefs & Fuse with Spatial Tokens
        belief_embeds = self.belief_proj(belief_probs)  # (B, 72, d_model)
        fused_board_tokens = torch.cat([board_tokens, belief_embeds], dim=-1)  # (B, 72, 2 * d_model)

        # 6. Policy Head (Explicitly conditioned on spatial state + belief distribution)
        policy_raw = self.policy_head(fused_board_tokens)  # (B, 72, 4)
        policy_flat = policy_raw.view(B, -1)               # (B, 288)

        # Action masking: set illegal actions to -1e4 (safe for bfloat16/float32)
        policy_logits = torch.where(action_mask, policy_flat, torch.full_like(policy_flat, -1e4))

        # 7. Value Head (Conditioned on [CLS] + global belief summary)
        fused_cls = torch.cat([cls_out, belief_embeds.mean(dim=1)], dim=-1)  # (B, 2 * d_model)
        value = self.value_head(fused_cls)  # (B, 1) in [-1, 1]

        return ModelOutput(
            policy_logits=policy_logits,
            value=value,
            belief_probs=belief_probs,
            belief_log_probs=belief_log_probs,
            sinkhorn_loss=sinkhorn_loss,
        )

    def get_action_and_value(
        self,
        piece_tokens: torch.Tensor,
        temporal_features: torch.Tensor,
        action_mask: torch.Tensor,
        enemy_alive_counts: torch.Tensor,
        revealed_mask: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        true_enemy_ranks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Inference and PPO loss helper.
        Returns:
            action: (B,) sampled or provided action
            log_prob: (B,) log probability of chosen action
            entropy: (B,) policy distribution entropy
            value: (B, 1) scalar value estimate
            sinkhorn_loss: scalar belief auxiliary loss
        """
        output = self.forward(
            piece_tokens=piece_tokens,
            temporal_features=temporal_features,
            action_mask=action_mask,
            enemy_alive_counts=enemy_alive_counts,
            revealed_mask=revealed_mask,
            true_enemy_ranks=true_enemy_ranks,
        )

        dist = torch.distributions.Categorical(logits=output.policy_logits)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, log_prob, entropy, output.value, output.sinkhorn_loss
