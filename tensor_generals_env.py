"""
tensor_generals_env.py

Fully vectorized, GPU-resident PyTorch environment for 'Game of the Generals' (Salpakan).
Simulates B parallel environments concurrently on CUDA with zero CPU transfers.

Features:
- Board: 9 rows x 8 cols = 72 squares.
- 15 Ranks: 0 (Flag) through 14 (5-Star General), 21 pieces per player.
- Immutable static combat tensor lookup COMBAT_LOOKUP[15, 15].
- Canonical frame: Player 2 rotated 180 deg (torch.rot90(x, 2, dims=(-2, -1))),
  with exact direction inversion (canonical_dir ^ 1).
- Categorical(288) action space with legal action boolean masking.
- Per-square behavioral history and temporal context (B, 72, 8).
- Auto-reset on termination (Flag capture / backline reach) and truncation (40 plies no-combat / 200 steps).
"""

from typing import Dict, NamedTuple, Optional, Tuple
import torch
import torch.nn.functional as F


# Piece Ranks and Army Composition
NUM_RANKS = 15
TOTAL_PIECES_PER_PLAYER = 21
NUM_ROWS = 9
NUM_COLS = 8
NUM_SQUARES = 72  # 9 * 8
NUM_DIRECTIONS = 4  # 0: North, 1: South, 2: West, 3: East
ACTION_SPACE_SIZE = 288  # 72 * 4

# Direction Deltas (dr, dc)
DIR_DELTAS = [
    (-1, 0),  # 0: North (Row - 1)
    (1, 0),   # 1: South (Row + 1)
    (0, -1),  # 2: West  (Col - 1)
    (0, 1),   # 3: East  (Col + 1)
]

# Starting multiset counts for ranks 0..14
STARTING_PIECE_COUNTS = [
    1,  # Rank 0: Flag
    2,  # Rank 1: Spy
    6,  # Rank 2: Private
    1,  # Rank 3: Sergeant
    1,  # Rank 4: 2nd Lieutenant
    1,  # Rank 5: 1st Lieutenant
    1,  # Rank 6: Captain
    1,  # Rank 7: Major
    1,  # Rank 8: Lt. Colonel
    1,  # Rank 9: Colonel
    1,  # Rank 10: 1-Star General
    1,  # Rank 11: 2-Star General
    1,  # Rank 12: 3-Star General
    1,  # Rank 13: 4-Star General
    1,  # Rank 14: 5-Star General
]


def build_combat_lookup(device: torch.device) -> torch.Tensor:
    """
    Constructs the immutable (15, 15) static tensor for combat resolution.
    Returns:
        +1: Attacker wins (defender removed, attacker advances)
        -1: Defender wins (attacker removed, defender holds square)
         0: Mutual destruction (both pieces removed)
    """
    lookup = torch.zeros((15, 15), dtype=torch.int8, device=device)
    for att in range(15):
        for defense in range(15):
            if att == defense:
                if att == 0:
                    # Flag vs Flag: attacking Flag eliminates defending Flag
                    lookup[att, defense] = 1
                else:
                    # Equal rank mutual elimination
                    lookup[att, defense] = 0
            elif att == 0:
                # Flag attacking any non-flag piece loses
                lookup[att, defense] = -1
            elif defense == 0:
                # Any non-flag piece attacking Flag wins
                lookup[att, defense] = 1
            elif att == 1:
                # Attacking Spy vs Private loses, vs Officers (3..14) wins
                lookup[att, defense] = -1 if defense == 2 else 1
            elif defense == 1:
                # Defending Spy vs Private loses, vs Officers (3..14) wins
                lookup[att, defense] = 1 if att == 2 else -1
            elif att == 2:
                # Attacking Private vs Spy wins, vs Officers (3..14) loses
                lookup[att, defense] = 1 if defense == 1 else -1
            elif defense == 2:
                # Defending Private vs Spy wins, vs Officers (3..14) loses
                lookup[att, defense] = -1 if att == 1 else 1
            else:
                # Officer vs Officer (3..14)
                lookup[att, defense] = 1 if att > defense else -1
    return lookup


class EnvObservation(NamedTuple):
    piece_tokens: torch.Tensor       # (B, 72) int64 in [0..16]
    temporal_features: torch.Tensor  # (B, 72, 8) float32
    action_mask: torch.Tensor        # (B, 288) bool
    current_player: torch.Tensor     # (B,) int64 (0 or 1)
    enemy_alive_counts: torch.Tensor # (B, 15) float32 remaining enemy piece counts per rank
    true_enemy_ranks: torch.Tensor   # (B, 72) int64 privileged true enemy ranks (-1 if not enemy)
    revealed_mask: torch.Tensor      # (B, 72) bool whether enemy on this square is revealed


class TensorGeneralsEnv:
    """
    Batched, 100% CUDA-vectorized environment for Game of the Generals (Salpakan).
    """

    def __init__(
        self,
        num_envs: int = 512,
        device: str = "cuda",
        max_steps: int = 200,
        max_plies_no_combat: int = 40,
    ) -> None:
        self.num_envs = num_envs
        self.device = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
        self.max_steps = max_steps
        self.max_plies_no_combat = max_plies_no_combat

        # Precompute static tensors
        self.combat_lookup = build_combat_lookup(self.device)
        self.starting_counts_tensor = torch.tensor(STARTING_PIECE_COUNTS, dtype=torch.int32, device=self.device)

        # Precompute board coordinate transitions and legal destination lookup
        # Shape: (72, 4) -> to_sq (or -1 if out of bounds)
        self.transition_matrix = torch.full((NUM_SQUARES, NUM_DIRECTIONS), -1, dtype=torch.int64, device=self.device)
        for sq in range(NUM_SQUARES):
            r, c = sq // NUM_COLS, sq % NUM_COLS
            for d, (dr, dc) in enumerate(DIR_DELTAS):
                nr, nc = r + dr, c + dc
                if 0 <= nr < NUM_ROWS and 0 <= nc < NUM_COLS:
                    self.transition_matrix[sq, d] = nr * NUM_COLS + nc

        # Canonical direction inversion tensor: [1, 0, 3, 2] (N<->S, W<->E)
        self.dir_inversion = torch.tensor([1, 0, 3, 2], dtype=torch.int64, device=self.device)

        # Precompute standard piece multiset list for placement
        piece_list = []
        for rank, count in enumerate(STARTING_PIECE_COUNTS):
            piece_list.extend([rank] * count)
        self.piece_multiset = torch.tensor(piece_list, dtype=torch.int64, device=self.device)  # length 21

        # Allocate simulation state tensors permanently on device
        self.board_pieces = torch.full((self.num_envs, NUM_SQUARES), -1, dtype=torch.int64, device=self.device)
        self.board_owners = torch.full((self.num_envs, NUM_SQUARES), -1, dtype=torch.int8, device=self.device)
        self.piece_alive_counts = torch.zeros((self.num_envs, 2, NUM_RANKS), dtype=torch.int32, device=self.device)
        self.current_player = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)

        # Per-square behavioral history
        self.moves_count = torch.zeros((self.num_envs, NUM_SQUARES), dtype=torch.float32, device=self.device)
        self.plies_since_moved = torch.zeros((self.num_envs, NUM_SQUARES), dtype=torch.float32, device=self.device)
        self.is_revealed = torch.zeros((self.num_envs, NUM_SQUARES), dtype=torch.float32, device=self.device)
        self.combats_survived = torch.zeros((self.num_envs, NUM_SQUARES), dtype=torch.float32, device=self.device)

        # Episode progression counters
        self.step_counts = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.plies_no_combat = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)

        # Initial reset
        env_mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.reset_envs(env_mask)

    def reset_envs(self, mask: torch.Tensor) -> None:
        """
        Resets environments specified by boolean mask.
        Generates randomized valid deployment across the 24 home squares for each player.
        """
        num_reset = int(mask.sum().item())
        if num_reset == 0:
            return

        idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)

        # Reset counters
        self.board_pieces[idx] = -1
        self.board_owners[idx] = -1
        self.current_player[idx] = 0
        self.moves_count[idx] = 0.0
        self.plies_since_moved[idx] = 0.0
        self.is_revealed[idx] = 0.0
        self.combats_survived[idx] = 0.0
        self.step_counts[idx] = 0
        self.plies_no_combat[idx] = 0

        # Reset piece counts: (num_reset, 2, 15)
        self.piece_alive_counts[idx, 0] = self.starting_counts_tensor.unsqueeze(0).expand(num_reset, -1)
        self.piece_alive_counts[idx, 1] = self.starting_counts_tensor.unsqueeze(0).expand(num_reset, -1)

        # Deploy Player 1 pieces across squares 0..23 (Rows 0..2)
        # 21 pieces + 3 empty (-1)
        p1_pool = torch.cat([self.piece_multiset, torch.full((3,), -1, dtype=torch.int64, device=self.device)])
        p1_batches = p1_pool.unsqueeze(0).expand(num_reset, 24)
        p1_perms = torch.rand(num_reset, 24, device=self.device).argsort(dim=-1)
        p1_placed = torch.gather(p1_batches, 1, p1_perms)

        self.board_pieces[idx, 0:24] = p1_placed
        p1_has_piece = p1_placed != -1
        self.board_owners[idx, 0:24] = torch.where(
            p1_has_piece,
            torch.zeros_like(p1_placed, dtype=torch.int8),
            torch.full_like(p1_placed, -1, dtype=torch.int8)
        )

        # Deploy Player 2 pieces across squares 48..71 (Rows 6..8)
        p2_pool = torch.cat([self.piece_multiset, torch.full((3,), -1, dtype=torch.int64, device=self.device)])
        p2_batches = p2_pool.unsqueeze(0).expand(num_reset, 24)
        p2_perms = torch.rand(num_reset, 24, device=self.device).argsort(dim=-1)
        p2_placed = torch.gather(p2_batches, 1, p2_perms)

        self.board_pieces[idx, 48:72] = p2_placed
        p2_has_piece = p2_placed != -1
        self.board_owners[idx, 48:72] = torch.where(
            p2_has_piece,
            torch.ones_like(p2_placed, dtype=torch.int8),
            torch.full_like(p2_placed, -1, dtype=torch.int8)
        )

    def get_action_mask(self, canonical: bool = True) -> torch.Tensor:
        """
        Computes legal action boolean mask of shape (B, 288).
        If canonical is True, returns mask from active player's canonical perspective.
        """
        # Absolute legal action mask
        b_idx = torch.arange(self.num_envs, device=self.device).unsqueeze(1).unsqueeze(2)  # (B, 1, 1)
        from_sqs = torch.arange(NUM_SQUARES, device=self.device).unsqueeze(0).unsqueeze(2)  # (1, 72, 1)
        dirs = torch.arange(NUM_DIRECTIONS, device=self.device).unsqueeze(0).unsqueeze(1)    # (1, 1, 4)

        to_sqs = self.transition_matrix[from_sqs, dirs]  # (1, 72, 4) broadcast -> (B, 72, 4)

        # Condition 1: Origin square has active player's piece
        active_player = self.current_player.unsqueeze(1).unsqueeze(2)  # (B, 1, 1)
        origin_owners = self.board_owners.unsqueeze(2)                 # (B, 72, 1)
        has_friendly_piece = (origin_owners == active_player)         # (B, 72, 1)

        # Condition 2: Destination is within board bounds
        in_bounds = (to_sqs >= 0)  # (1, 72, 4)

        # Condition 3: Destination square does NOT have a friendly piece
        safe_to_sqs = torch.clamp(to_sqs, min=0)
        dest_owners = torch.gather(self.board_owners.unsqueeze(1).expand(-1, 72, -1), 2, safe_to_sqs) # (B, 72, 4)
        not_friendly_dest = (dest_owners != active_player)

        abs_mask = (has_friendly_piece & in_bounds & not_friendly_dest).view(self.num_envs, ACTION_SPACE_SIZE)

        if not canonical:
            return abs_mask

        # Transform to canonical mask for Player 2
        # Player 1 is already canonical. For Player 2: canonical_sq = 71 - abs_sq, canonical_dir = abs_dir ^ 1
        is_p2 = (self.current_player == 1)
        if not is_p2.any():
            return abs_mask

        canonical_mask = abs_mask.clone()
        if is_p2.any():
            # Map canonical (sq, d) to absolute (71 - sq, d ^ 1)
            p2_indices = torch.nonzero(is_p2, as_tuple=False).squeeze(-1)
            # Reshape to (B, 72, 4)
            abs_reshaped = abs_mask[p2_indices].view(-1, NUM_SQUARES, NUM_DIRECTIONS)
            # Invert directions: [0, 1, 2, 3] -> [1, 0, 3, 2]
            abs_dir_swapped = abs_reshaped[:, :, self.dir_inversion]
            # Invert squares: 71 - sq
            canonical_p2 = torch.flip(abs_dir_swapped, dims=[1]).view(-1, ACTION_SPACE_SIZE)
            canonical_mask[p2_indices] = canonical_p2

        return canonical_mask

    def canonical_action_to_absolute(self, actions: torch.Tensor, player: torch.Tensor) -> torch.Tensor:
        """
        Converts canonical actions (from active player's viewpoint) to absolute board actions.
        Canonical action: a in [0..287] -> sq = a // 4, d = a % 4
        For Player 1: abs_sq = sq, abs_d = d
        For Player 2: abs_sq = 71 - sq, abs_d = d ^ 1
        """
        sq = actions // NUM_DIRECTIONS
        d = actions % NUM_DIRECTIONS

        p2_mask = (player == 1)
        abs_sq = torch.where(p2_mask, (NUM_SQUARES - 1) - sq, sq)
        abs_d = torch.where(p2_mask, d ^ 1, d)

        return abs_sq * NUM_DIRECTIONS + abs_d

    def get_canonical_observation(self) -> EnvObservation:
        """
        Generates canonical observations for the active player.
        Player 2 observations are rotated 180 degrees.
        """
        b_range = torch.arange(self.num_envs, device=self.device)
        active_p = self.current_player
        opp_p = 1 - active_p

        # 1. Piece Tokens (17 categories: 0=Empty, 1..15=Own pieces, 16=Enemy piece)
        is_own = (self.board_owners == active_p.unsqueeze(1))
        is_opp = (self.board_owners == opp_p.unsqueeze(1))

        # Pieces representation:
        # Own piece rank 0..14 -> token 1..15
        # Opp piece -> token 16
        # Empty -> token 0
        raw_tokens = torch.zeros((self.num_envs, NUM_SQUARES), dtype=torch.int64, device=self.device)
        raw_tokens = torch.where(is_own, self.board_pieces + 1, raw_tokens)
        raw_tokens = torch.where(is_opp, torch.full_like(raw_tokens, 16), raw_tokens)

        # 2. Behavioral and Temporal Features (8 dims per square)
        # f0: moves_count / 20.0
        # f1: plies_since_moved / 40.0
        # f2: is_revealed
        # f3: combats_survived / 5.0
        # f4: step_counts / 200.0 (global)
        # f5: plies_no_combat / 40.0 (global)
        # f6: own_pieces_alive / 21.0 (global)
        # f7: opp_pieces_alive / 21.0 (global)

        own_alive = self.piece_alive_counts[b_range, active_p].sum(dim=-1, keepdim=True).float() / float(TOTAL_PIECES_PER_PLAYER)
        opp_alive = self.piece_alive_counts[b_range, opp_p].sum(dim=-1, keepdim=True).float() / float(TOTAL_PIECES_PER_PLAYER)
        steps_norm = (self.step_counts.float() / float(self.max_steps)).unsqueeze(1)
        no_combat_norm = (self.plies_no_combat.float() / float(self.max_plies_no_combat)).unsqueeze(1)

        f0 = (self.moves_count / 20.0).unsqueeze(-1)
        f1 = (self.plies_since_moved / 40.0).unsqueeze(-1)
        f2 = self.is_revealed.unsqueeze(-1)
        f3 = (self.combats_survived / 5.0).unsqueeze(-1)
        f4 = steps_norm.unsqueeze(1).expand(-1, NUM_SQUARES, 1)
        f5 = no_combat_norm.unsqueeze(1).expand(-1, NUM_SQUARES, 1)
        f6 = own_alive.unsqueeze(1).expand(-1, NUM_SQUARES, 1)
        f7 = opp_alive.unsqueeze(1).expand(-1, NUM_SQUARES, 1)

        raw_features = torch.cat([f0, f1, f2, f3, f4, f5, f6, f7], dim=-1)  # (B, 72, 8)

        # Privileged true enemy ranks (for Sinkhorn belief loss computation)
        raw_enemy_ranks = torch.where(is_opp, self.board_pieces, torch.full_like(self.board_pieces, -1))
        raw_revealed_mask = (is_opp & (self.is_revealed > 0.5))

        # Rotate 180 degrees for Player 2
        is_p2 = (active_p == 1)
        if is_p2.any():
            p2_idx = torch.nonzero(is_p2, as_tuple=False).squeeze(-1)

            # 180 degree rotation reverses square indexing (71 - sq)
            raw_tokens[p2_idx] = torch.flip(raw_tokens[p2_idx], dims=[1])
            raw_features[p2_idx] = torch.flip(raw_features[p2_idx], dims=[1])
            raw_enemy_ranks[p2_idx] = torch.flip(raw_enemy_ranks[p2_idx], dims=[1])
            raw_revealed_mask[p2_idx] = torch.flip(raw_revealed_mask[p2_idx], dims=[1])

        # Enemy surviving counts per rank for active player's Sinkhorn belief target
        enemy_alive = self.piece_alive_counts[b_range, opp_p].float()  # (B, 15)

        action_mask = self.get_action_mask(canonical=True)

        return EnvObservation(
            piece_tokens=raw_tokens,
            temporal_features=raw_features,
            action_mask=action_mask,
            current_player=active_p,
            enemy_alive_counts=enemy_alive,
            true_enemy_ranks=raw_enemy_ranks,
            revealed_mask=raw_revealed_mask,
        )

    def is_flag_unthreatened_at_backline(
        self,
        flag_positions: torch.Tensor,
        active_player: torch.Tensor,
        active_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Vectorized check whether active player's Flag has reached opponent backline unthreatened.
        Backline for P1: Row 8 (squares 64..71).
        Backline for P2: Row 0 (squares 0..7).
        Unthreatened: No enemy piece orthogonally adjacent to flag square.
        """
        unthreatened = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if not active_mask.any():
            return unthreatened

        env_indices = torch.nonzero(active_mask, as_tuple=False).squeeze(-1)
        sqs = flag_positions[env_indices]
        p = active_player[env_indices]
        opp = 1 - p

        # Check if at backline
        rows = sqs // NUM_COLS
        at_backline = torch.where(p == 0, rows == (NUM_ROWS - 1), rows == 0)

        if not at_backline.any():
            return unthreatened

        backline_envs = env_indices[at_backline]
        backline_sqs = sqs[at_backline]
        backline_opp = opp[at_backline]

        # Check adjacent 4 squares for enemy piece
        adj_threat = torch.zeros(backline_envs.shape[0], dtype=torch.bool, device=self.device)
        for d in range(NUM_DIRECTIONS):
            neighbor_sqs = self.transition_matrix[backline_sqs, d]
            valid_neighbor = neighbor_sqs >= 0
            safe_neighbor = torch.clamp(neighbor_sqs, min=0)

            neighbor_owners = self.board_owners[backline_envs, safe_neighbor]
            has_enemy = valid_neighbor & (neighbor_owners == backline_opp)
            adj_threat = adj_threat | has_enemy

        backline_safe = ~adj_threat
        unthreatened[backline_envs[backline_safe]] = True
        return unthreatened

    def step(
        self,
        canonical_actions: torch.Tensor
    ) -> Tuple[EnvObservation, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Executes one environmental step concurrently across all B environments.
        
        Args:
            canonical_actions: (B,) actions in active player's canonical coordinate space.
            
        Returns:
            next_obs: Canonical observation for the next active player.
            reward: (B,) reward for the player who just acted (+1 win, -1 loss, 0 otherwise).
            terminated: (B,) boolean terminal flag (Flag captured or backline reached).
            truncated: (B,) boolean truncation flag (40 plies no-combat or 200 steps).
            info: Dictionary containing winner info, combat indicators, etc.
        """
        b_range = torch.arange(self.num_envs, device=self.device)
        acting_p = self.current_player
        opp_p = 1 - acting_p

        # 1. Map canonical actions to absolute actions
        abs_actions = self.canonical_action_to_absolute(canonical_actions, acting_p)
        from_sqs = abs_actions // NUM_DIRECTIONS
        dirs = abs_actions % NUM_DIRECTIONS
        to_sqs = self.transition_matrix[from_sqs, dirs]

        # 2. Update plies since moved for all pieces
        self.plies_since_moved += 1.0
        self.step_counts += 1

        # Track active piece ranks and destination state
        moving_ranks = self.board_pieces[b_range, from_sqs]
        dest_ranks = self.board_pieces[b_range, torch.clamp(to_sqs, min=0)]
        dest_owners = self.board_owners[b_range, torch.clamp(to_sqs, min=0)]

        is_combat = (dest_owners == opp_p)
        is_empty = (dest_owners == -1)

        # Rewards and flags
        rewards = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        truncated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Case A: Move into empty square
        empty_mask = is_empty & (to_sqs >= 0)
        if empty_mask.any():
            e_idx = torch.nonzero(empty_mask, as_tuple=False).squeeze(-1)
            e_from = from_sqs[e_idx]
            e_to = to_sqs[e_idx]
            e_p = acting_p[e_idx]
            e_ranks = moving_ranks[e_idx]

            # Transfer piece
            self.board_pieces[e_idx, e_to] = e_ranks
            self.board_owners[e_idx, e_to] = e_p.to(torch.int8)
            self.board_pieces[e_idx, e_from] = -1
            self.board_owners[e_idx, e_from] = -1

            # Update behavioral history
            self.moves_count[e_idx, e_to] = self.moves_count[e_idx, e_from] + 1.0
            self.plies_since_moved[e_idx, e_to] = 0.0
            self.is_revealed[e_idx, e_to] = self.is_revealed[e_idx, e_from]
            self.combats_survived[e_idx, e_to] = self.combats_survived[e_idx, e_from]

            # Clear origin square history
            self.moves_count[e_idx, e_from] = 0.0
            self.plies_since_moved[e_idx, e_from] = 0.0
            self.is_revealed[e_idx, e_from] = 0.0
            self.combats_survived[e_idx, e_from] = 0.0

            self.plies_no_combat[e_idx] += 1

            # Check if Flag reached backline unthreatened
            is_flag = (e_ranks == 0)
            if is_flag.any():
                flag_envs = e_idx[is_flag]
                flag_to = e_to[is_flag]
                flag_p = e_p[is_flag]
                flag_active_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
                flag_active_mask[flag_envs] = True
                flag_pos_tensor = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
                flag_pos_tensor[flag_envs] = flag_to

                flag_wins = self.is_flag_unthreatened_at_backline(flag_pos_tensor, acting_p, flag_active_mask)
                if flag_wins.any():
                    w_idx = torch.nonzero(flag_wins, as_tuple=False).squeeze(-1)
                    rewards[w_idx] = 1.0
                    terminated[w_idx] = True

        # Case B: Combat encounter
        if is_combat.any():
            c_idx = torch.nonzero(is_combat, as_tuple=False).squeeze(-1)
            c_from = from_sqs[c_idx]
            c_to = to_sqs[c_idx]
            c_p = acting_p[c_idx]
            c_opp = opp_p[c_idx]
            c_att_rank = moving_ranks[c_idx]
            c_def_rank = dest_ranks[c_idx]

            # Reset no-combat plies
            self.plies_no_combat[c_idx] = 0

            # Static lookup outcome: +1 (att wins), -1 (def wins), 0 (mutual)
            outcome = self.combat_lookup[c_att_rank, c_def_rank]

            # B1: Attacker Wins (+1)
            att_win_mask = (outcome == 1)
            if att_win_mask.any():
                aw_idx = c_idx[att_win_mask]
                aw_from = c_from[att_win_mask]
                aw_to = c_to[att_win_mask]
                aw_p = c_p[att_win_mask]
                aw_opp = c_opp[att_win_mask]
                aw_att_rank = c_att_rank[att_win_mask]
                aw_def_rank = c_def_rank[att_win_mask]

                # Defender eliminated
                self.piece_alive_counts[aw_idx, aw_opp, aw_def_rank] -= 1

                # Attacker advances to destination
                self.board_pieces[aw_idx, aw_to] = aw_att_rank
                self.board_owners[aw_idx, aw_to] = aw_p.to(torch.int8)
                self.board_pieces[aw_idx, aw_from] = -1
                self.board_owners[aw_idx, aw_from] = -1

                # Attacker is revealed & survived combat
                self.moves_count[aw_idx, aw_to] = self.moves_count[aw_idx, aw_from] + 1.0
                self.plies_since_moved[aw_idx, aw_to] = 0.0
                self.is_revealed[aw_idx, aw_to] = 1.0
                self.combats_survived[aw_idx, aw_to] = self.combats_survived[aw_idx, aw_from] + 1.0

                # Clear origin
                self.moves_count[aw_idx, aw_from] = 0.0
                self.plies_since_moved[aw_idx, aw_from] = 0.0
                self.is_revealed[aw_idx, aw_from] = 0.0
                self.combats_survived[aw_idx, aw_from] = 0.0

                # Defender flag captured -> immediate win
                captured_flag = (aw_def_rank == 0)
                if captured_flag.any():
                    flag_win_idx = aw_idx[captured_flag]
                    rewards[flag_win_idx] = 1.0
                    terminated[flag_win_idx] = True

                # Attacker flag reached backline unthreatened
                is_att_flag = (aw_att_rank == 0) & (~captured_flag)
                if is_att_flag.any():
                    flag_envs = aw_idx[is_att_flag]
                    flag_to = aw_to[is_att_flag]
                    flag_active_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
                    flag_active_mask[flag_envs] = True
                    flag_pos_tensor = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
                    flag_pos_tensor[flag_envs] = flag_to

                    flag_wins = self.is_flag_unthreatened_at_backline(flag_pos_tensor, acting_p, flag_active_mask)
                    if flag_wins.any():
                        w_idx = torch.nonzero(flag_wins, as_tuple=False).squeeze(-1)
                        rewards[w_idx] = 1.0
                        terminated[w_idx] = True

            # B2: Defender Wins (-1)
            def_win_mask = (outcome == -1)
            if def_win_mask.any():
                dw_idx = c_idx[def_win_mask]
                dw_from = c_from[def_win_mask]
                dw_to = c_to[def_win_mask]
                dw_p = c_p[def_win_mask]
                dw_att_rank = c_att_rank[def_win_mask]

                # Attacker eliminated
                self.piece_alive_counts[dw_idx, dw_p, dw_att_rank] -= 1

                # Clear attacker origin
                self.board_pieces[dw_idx, dw_from] = -1
                self.board_owners[dw_idx, dw_from] = -1
                self.moves_count[dw_idx, dw_from] = 0.0
                self.plies_since_moved[dw_idx, dw_from] = 0.0
                self.is_revealed[dw_idx, dw_from] = 0.0
                self.combats_survived[dw_idx, dw_from] = 0.0

                # Defender holds square, revealed to attacker, combats survived + 1
                self.is_revealed[dw_idx, dw_to] = 1.0
                self.combats_survived[dw_idx, dw_to] += 1.0

                # If attacker was Flag, active player lost
                att_lost_flag = (dw_att_rank == 0)
                if att_lost_flag.any():
                    flag_lost_idx = dw_idx[att_lost_flag]
                    rewards[flag_lost_idx] = -1.0
                    terminated[flag_lost_idx] = True

            # B3: Mutual Elimination (0)
            mutual_mask = (outcome == 0)
            if mutual_mask.any():
                m_idx = c_idx[mutual_mask]
                m_from = c_from[mutual_mask]
                m_to = c_to[mutual_mask]
                m_p = c_p[mutual_mask]
                m_opp = c_opp[mutual_mask]
                m_att_rank = c_att_rank[mutual_mask]
                m_def_rank = c_def_rank[mutual_mask]

                # Both pieces eliminated
                self.piece_alive_counts[m_idx, m_p, m_att_rank] -= 1
                self.piece_alive_counts[m_idx, m_opp, m_def_rank] -= 1

                self.board_pieces[m_idx, m_from] = -1
                self.board_owners[m_idx, m_from] = -1
                self.board_pieces[m_idx, m_to] = -1
                self.board_owners[m_idx, m_to] = -1

                self.moves_count[m_idx, m_from] = 0.0
                self.plies_since_moved[m_idx, m_from] = 0.0
                self.is_revealed[m_idx, m_from] = 0.0
                self.combats_survived[m_idx, m_from] = 0.0

                self.moves_count[m_idx, m_to] = 0.0
                self.plies_since_moved[m_idx, m_to] = 0.0
                self.is_revealed[m_idx, m_to] = 0.0
                self.combats_survived[m_idx, m_to] = 0.0

        # Check Truncation: 40 plies without combat or step limit reached
        trunc_no_combat = (self.plies_no_combat >= self.max_plies_no_combat)
        trunc_steps = (self.step_counts >= self.max_steps)
        truncated = (trunc_no_combat | trunc_steps) & (~terminated)

        # Switch active player for non-terminal environments
        self.current_player = 1 - self.current_player

        # Check if next player has any legal moves; if not, current acting player wins
        next_mask = self.get_action_mask(canonical=False)
        no_moves = (next_mask.sum(dim=-1) == 0) & (~terminated) & (~truncated)
        if no_moves.any():
            nm_idx = torch.nonzero(no_moves, as_tuple=False).squeeze(-1)
            rewards[nm_idx] = 1.0
            terminated[nm_idx] = True

        done = terminated | truncated

        info = {
            "is_combat": is_combat,
            "acting_player": acting_p,
            "terminated": terminated,
            "truncated": truncated,
        }

        # Auto-reset completed environments seamlessly
        if done.any():
            self.reset_envs(done)

        next_obs = self.get_canonical_observation()
        return next_obs, rewards, terminated, truncated, info
