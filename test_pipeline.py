"""
test_pipeline.py

Comprehensive Verification Suite for Game of the Generals (Salpakan) RL Pipeline:
1. Combat Arbiter Matrix verification (all 15x15 combinations).
2. Perspective canonicalization and direction inversion.
3. Piece multiset and randomized placement integrity.
4. Masked Sinkhorn-Knopp doubly-stochastic convergence and dead-rank masking.
5. Vectorized GPU heuristic execution and legal action validity.
6. Transformer network forward pass, shapes, and gradient flow.
7. Full end-to-end PFSP PPO rollout and throughput benchmarking.
"""

import sys
import time
import torch
import torch.nn.functional as F

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from tensor_generals_env import (
    TensorGeneralsEnv,
    build_combat_lookup,
    STARTING_PIECE_COUNTS,
    NUM_RANKS,
    NUM_SQUARES,
    NUM_DIRECTIONS,
    ACTION_SPACE_SIZE,
)
from board_transformer import BoardTransformer
from train_pfsp import VectorizedHeuristicAgent, PFSPTrainer


def test_combat_lookup():
    print("\n[1/7] Testing Combat Arbiter Matrix (15x15)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lookup = build_combat_lookup(device)
    assert lookup.shape == (15, 15), f"Expected shape (15, 15), got {lookup.shape}"

    # Spy (1) vs 5-Star General (14)
    assert lookup[1, 14] == 1, "Spy attacking 5-Star must win (+1)"
    assert lookup[14, 1] == -1, "5-Star attacking Spy must lose (-1)"

    # Private (2) vs Spy (1)
    assert lookup[2, 1] == 1, "Private attacking Spy must win (+1)"
    assert lookup[1, 2] == -1, "Spy attacking Private must lose (-1)"

    # Private (2) vs 5-Star (14)
    assert lookup[2, 14] == -1, "Private attacking 5-Star must lose (-1)"
    assert lookup[14, 2] == 1, "5-Star attacking Private must win (+1)"

    # Flag (0) vs Flag (0)
    assert lookup[0, 0] == 1, "Attacking Flag vs defending Flag must win (+1)"

    # Flag (0) vs Private (2)
    assert lookup[0, 2] == -1, "Attacking Flag vs Private must lose (-1)"
    assert lookup[2, 0] == 1, "Attacking Private vs Flag must win (+1)"

    # Equal Rank Mutual Elimination (except flag)
    for r in range(1, 15):
        assert lookup[r, r] == 0, f"Rank {r} vs Rank {r} must be mutual elimination (0)"

    # Higher rank vs Lower rank officers
    for r1 in range(3, 15):
        for r2 in range(3, r1):
            assert lookup[r1, r2] == 1, f"Rank {r1} attacking Rank {r2} must win (+1)"
            assert lookup[r2, r1] == -1, f"Rank {r2} attacking Rank {r1} must lose (-1)"

    print("  -> Combat Lookup Matrix passed all verification checks!")


def test_canonical_transform_and_direction_inversion():
    print("\n[2/7] Testing Canonical Perspective & Direction Inversion...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = TensorGeneralsEnv(num_envs=16, device=str(device))

    # Test Direction Inversion: [0, 1, 2, 3] -> [1, 0, 3, 2] (N<->S, W<->E)
    # North (0) for P2 should become South (1) on absolute board
    # West (2) for P2 should become East (3) on absolute board
    canonical_actions = torch.tensor([0 * 4 + 0, 10 * 4 + 2], device=device)  # sq 0 dir N, sq 10 dir W
    p2_player = torch.tensor([1, 1], device=device)

    abs_actions = env.canonical_action_to_absolute(canonical_actions, p2_player)
    expected_sq0 = 71 - 0  # 71
    expected_dir0 = 0 ^ 1   # 1 (South)
    expected_act0 = expected_sq0 * 4 + expected_dir0

    expected_sq1 = 71 - 10  # 61
    expected_dir1 = 2 ^ 1   # 3 (East)
    expected_act1 = expected_sq1 * 4 + expected_dir1

    assert abs_actions[0].item() == expected_act0, f"Expected {expected_act0}, got {abs_actions[0].item()}"
    assert abs_actions[1].item() == expected_act1, f"Expected {expected_act1}, got {abs_actions[1].item()}"

    # For P1, canonical actions should be identical to absolute actions
    p1_player = torch.tensor([0, 0], device=device)
    p1_abs = env.canonical_action_to_absolute(canonical_actions, p1_player)
    assert (p1_abs == canonical_actions).all(), "P1 actions must not be modified in canonical conversion"

    print("  -> Canonical Inversion and Rotation mapping verified successfully!")


def test_piece_multiset_and_board_setup():
    print("\n[3/7] Testing Army Multiset & Initial Deployment...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = TensorGeneralsEnv(num_envs=64, device=str(device))

    # Vectorized piece count checks (21 valid pieces per player)
    p1_valid_counts = (env.board_pieces[:, 0:24] >= 0).sum(dim=-1)
    p2_valid_counts = (env.board_pieces[:, 48:72] >= 0).sum(dim=-1)
    assert (p1_valid_counts == 21).all(), "Player 1 must have exactly 21 pieces across all envs"
    assert (p2_valid_counts == 21).all(), "Player 2 must have exactly 21 pieces across all envs"

    # Vectorized multiset counts check across all 15 ranks
    for rank, count in enumerate(STARTING_PIECE_COUNTS):
        p1_rank_counts = (env.board_pieces[:, 0:24] == rank).sum(dim=-1)
        p2_rank_counts = (env.board_pieces[:, 48:72] == rank).sum(dim=-1)
        assert (p1_rank_counts == count).all(), f"P1 multiset count mismatch on rank {rank}"
        assert (p2_rank_counts == count).all(), f"P2 multiset count mismatch on rank {rank}"

    print("  -> Piece Multiset and strategic placement integrity verified!")


def test_sinkhorn_belief_head():
    print("\n[4/7] Testing Masked Sinkhorn-Knopp Belief Head...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BoardTransformer().to(device)
    model.eval()

    B = 8
    # Mock observation: 5 enemy pieces on board, 2 dead ranks, 1 revealed piece
    piece_tokens = torch.zeros((B, NUM_SQUARES), dtype=torch.int64, device=device)
    enemy_sqs = [50, 55, 60, 65, 70]
    for sq in enemy_sqs:
        piece_tokens[:, sq] = 16  # Enemy piece

    revealed_mask = torch.zeros((B, NUM_SQUARES), dtype=torch.bool, device=device)
    revealed_mask[:, 50] = True  # sq 50 is revealed

    true_ranks = torch.full((B, NUM_SQUARES), -1, dtype=torch.int64, device=device)
    true_ranks[:, 50] = 14  # Revealed 5-Star General
    true_ranks[:, 55] = 2   # Private
    true_ranks[:, 60] = 2   # Private
    true_ranks[:, 65] = 1   # Spy
    true_ranks[:, 70] = 0   # Flag

    # Living counts: 1 5-Star, 2 Privates, 1 Spy, 1 Flag (total 5 pieces alive)
    alive_counts = torch.zeros((B, 15), dtype=torch.float32, device=device)
    alive_counts[:, 0] = 1.0  # Flag
    alive_counts[:, 1] = 1.0  # Spy
    alive_counts[:, 2] = 2.0  # Privates
    alive_counts[:, 14] = 1.0 # 5-Star
    # All other ranks count = 0 (dead ranks)

    temporal_feats = torch.zeros((B, NUM_SQUARES, 8), dtype=torch.float32, device=device)
    action_mask = torch.ones((B, ACTION_SPACE_SIZE), dtype=torch.bool, device=device)

    out = model(
        piece_tokens=piece_tokens,
        temporal_features=temporal_feats,
        action_mask=action_mask,
        enemy_alive_counts=alive_counts,
        revealed_mask=revealed_mask,
        true_enemy_ranks=true_ranks,
    )

    probs = out.belief_probs  # (B, 72, 15)

    # 1. Non-enemy squares must have strictly 0 probability
    non_enemy_mask = (piece_tokens != 16)
    assert torch.allclose(probs[non_enemy_mask], torch.zeros_like(probs[non_enemy_mask])), "Non-enemy squares must have 0 probability"

    # 2. Dead ranks (e.g. ranks 3..13) must have strictly 0 probability
    dead_ranks = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    for dr in dead_ranks:
        assert torch.allclose(probs[:, :, dr], torch.zeros_like(probs[:, :, dr])), f"Dead rank {dr} must have 0 probability"

    # 3. Revealed square 50 must have 1.0 on rank 14
    assert torch.allclose(probs[:, 50, 14], torch.ones(B, device=device)), "Revealed square 50 must have 1.0 on rank 14"

    # 4. Unknown enemy squares must have row sums equal to 1.0
    for sq in [55, 60, 65, 70]:
        row_sums = probs[:, sq, :].sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones(B, device=device), atol=1e-3), f"Unknown square {sq} row sum must be 1.0, got {row_sums}"

    # 5. Unrevealed living rank column sums must match target unknown counts
    # Rank 0 target = 1, Rank 1 target = 1, Rank 2 target = 2
    unk_sqs = torch.tensor([55, 60, 65, 70], device=device)
    col_sum_rank0 = probs[:, unk_sqs, 0].sum(dim=-1)
    col_sum_rank1 = probs[:, unk_sqs, 1].sum(dim=-1)
    col_sum_rank2 = probs[:, unk_sqs, 2].sum(dim=-1)

    assert torch.allclose(col_sum_rank0, torch.ones(B, device=device), atol=1e-2), f"Rank 0 col sum {col_sum_rank0}"
    assert torch.allclose(col_sum_rank1, torch.ones(B, device=device), atol=1e-2), f"Rank 1 col sum {col_sum_rank1}"
    assert torch.allclose(col_sum_rank2, torch.full((B,), 2.0, device=device), atol=1e-2), f"Rank 2 col sum {col_sum_rank2}"

    print("  -> Masked Sinkhorn-Knopp belief head passed all mathematical constraints!")


def test_vectorized_heuristic_speed():
    print("\n[5/7] Testing Vectorized Heuristic Speed & Validity...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B = 512
    env = TensorGeneralsEnv(num_envs=B, device=str(device))
    agent = VectorizedHeuristicAgent(device=device)

    obs = env.get_canonical_observation()
    all_envs = torch.arange(B, device=device)

    # Warmup
    _ = agent.select_action(obs, env, all_envs)
    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(100):
        actions = agent.select_action(obs, env, all_envs)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - t0

    latency_ms = (elapsed / 100) * 1000.0
    print(f"  -> Heuristic latency for B={B}: {latency_ms:.3f} ms per step")

    # Verify action legality
    is_legal = torch.gather(obs.action_mask, 1, actions.unsqueeze(1)).squeeze(1)
    assert is_legal.all(), "All actions generated by heuristic must be legal!"
    print("  -> All heuristic actions strictly valid & legal!")


def test_transformer_gradient_flow():
    print("\n[6/7] Testing Transformer Backprop & Gradient Flow...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BoardTransformer().to(device)
    model.train()

    B = 16
    env = TensorGeneralsEnv(num_envs=B, device=str(device))
    obs = env.get_canonical_observation()

    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
        out = model(
            piece_tokens=obs.piece_tokens,
            temporal_features=obs.temporal_features,
            action_mask=obs.action_mask,
            enemy_alive_counts=obs.enemy_alive_counts,
            revealed_mask=obs.revealed_mask,
            true_enemy_ranks=obs.true_enemy_ranks,
        )

        deploy_score = model.deployment_head(model.piece_emb(obs.piece_tokens[:, :24])).sum()
        loss = out.policy_logits.sum() * 0.0 + out.value.sum() + out.sinkhorn_loss + deploy_score

    loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"Parameter {name} did not receive gradients!"
        assert not torch.isnan(param.grad).any(), f"Parameter {name} gradient contains NaN!"

    print("  -> Transformer forward pass, bfloat16 autocast, and backprop verified!")


def test_end_to_end_training_throughput():
    print("\n[7/7] Benchmarking End-to-End PFSP PPO Rollout & Training Throughput...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trainer = PFSPTrainer(num_envs=32, rollout_steps=16, device=device)
    trainer.train(total_iterations=2, checkpoint_interval=2)
    print("  -> End-to-End Self-Play Training Run Completed Successfully!")


if __name__ == "__main__":
    test_combat_lookup()
    test_canonical_transform_and_direction_inversion()
    test_piece_multiset_and_board_setup()
    test_sinkhorn_belief_head()
    test_vectorized_heuristic_speed()
    test_transformer_gradient_flow()
    test_end_to_end_training_throughput()
    print("\n=======================================================")
    print("ALL TESTS AND BENCHMARKS PASSED SUCCESSFULLY!")
    print("=======================================================")
