"""
test_features.py

Verification suite for all newly implemented extensions:
1. CheckpointManager save & restore integrity.
2. TrainingTelemetry & PitfallDetector alert detection.
3. GameRecorder trajectory capture & HTML replay rendering.
4. ResearchTracker YAML schema and Markdown changelog sync.
5. FastAPI Human vs AI backend endpoints test.
"""

import os
import shutil
import torch
from fastapi.testclient import TestClient

from checkpoint_manager import CheckpointManager
from training_telemetry import TrainingTelemetry, PitfallDetector
from game_observer import GameRecorder
from research_tracker import ResearchTracker
from board_transformer import BoardTransformer
from tensor_generals_env import TensorGeneralsEnv
from app import app


def test_checkpoint_manager():
    print("\n[1/5] Testing CheckpointManager Save & Restore...")
    test_dir = "test_checkpoints"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BoardTransformer().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    manager = CheckpointManager(checkpoint_dir=test_dir, max_rolling=3, max_best=2)

    # Save 5 rolling checkpoints
    for i in range(1, 6):
        saved = manager.save_checkpoint(
            iteration=i,
            model=model,
            optimizer=optimizer,
            scheduler=None,
            scaler=None,
            pfsp_pool=None,
            metrics={"loss_policy": 0.1 / i},
            score=i * 1.5,
        )
        assert os.path.exists(saved), f"Checkpoint file {saved} should exist"

    # Verify rolling pruning kept only latest 3
    rolling_files = [f for f in os.listdir(test_dir) if f.startswith("checkpoint_iter_")]
    assert len(rolling_files) == 3, f"Expected 3 rolling checkpoints, found {len(rolling_files)}"

    # Test Restore into fresh model
    fresh_model = BoardTransformer().to(device)
    fresh_opt = torch.optim.Adam(fresh_model.parameters(), lr=1e-3)
    loaded_iter, metrics = manager.load_checkpoint(
        checkpoint_path=os.path.join(test_dir, "latest.pt"),
        model=fresh_model,
        optimizer=fresh_opt,
        device=device,
    )
    assert loaded_iter == 5, f"Expected restored iter 5, got {loaded_iter}"

    # Clean up test dir
    shutil.rmtree(test_dir)
    print("  -> CheckpointManager save, prune, and restoration verified successfully!")


def test_pitfall_detector():
    print("\n[2/5] Testing TrainingTelemetry Pitfall Diagnostics...")
    detector = PitfallDetector()

    # Case 1: Healthy metrics
    healthy_alerts = detector.analyze({
        "loss_policy": 0.02,
        "loss_value": 0.01,
        "loss_sinkhorn": 2.1,
        "loss_entropy": 2.5,
        "combats_per_rollout": 1200.0,
        "terminals_per_rollout": 400.0,
        "truncations_per_rollout": 50.0,
    })
    assert len(healthy_alerts) == 0, "Healthy training should produce 0 alerts"

    # Case 2: Injected Entropy Collapse
    collapse_alerts = detector.analyze({
        "loss_policy": 0.01,
        "loss_value": 0.01,
        "loss_sinkhorn": 2.0,
        "loss_entropy": 0.20,  # Critical collapse
        "combats_per_rollout": 1000.0,
        "terminals_per_rollout": 300.0,
        "truncations_per_rollout": 20.0,
    })
    assert any("Entropy Collapse" in str(a) for a in collapse_alerts), "Should flag Entropy Collapse alert"

    # Case 3: Injected Turtling / Passive Stagnation
    turtling_alerts = detector.analyze({
        "loss_policy": 0.01,
        "loss_value": 0.01,
        "loss_sinkhorn": 2.0,
        "loss_entropy": 2.0,
        "combats_per_rollout": 10.0,  # Low combat
        "terminals_per_rollout": 10.0,
        "truncations_per_rollout": 90.0, # High truncation
    })
    assert any("Turtling" in str(a) for a in turtling_alerts), "Should flag Turtling alert"

    print("  -> Pitfall Diagnostics correctly caught simulated RL failure modes!")


def test_game_recorder_html():
    print("\n[3/5] Testing GameRecorder & HTML Replay Generation...")
    recorder = GameRecorder(p1_name="AlphaGenerals", p2_name="GreedyHeuristic")
    recorder.set_initial_state(
        pieces=[0] * 72,
        owners=[0] * 24 + [-1] * 24 + [1] * 24,
    )

    for step in range(1, 10):
        recorder.record_step(
            ply=step,
            acting_player=step % 2,
            from_sq=step,
            direction=1,
            to_sq=step + 8,
            moving_rank=2,
            dest_owner_before=-1,
            dest_rank_before=-1,
            combat_outcome=None if step < 8 else 1,
            board_pieces=[0] * 72,
            board_owners=[0] * 24 + [-1] * 24 + [1] * 24,
            is_revealed=[0.0] * 72,
            terminated=(step == 9),
            winner=0 if step == 9 else None,
        )

    out_file = "test_replay.html"
    recorder.export_html(out_file)
    assert os.path.isfile(out_file), "Replay file should be created"
    assert os.path.getsize(out_file) > 1000, "Replay HTML should contain full bundle"
    os.remove(out_file)
    print("  -> GameRecorder trajectory and interactive HTML replay verified!")


def test_research_tracker():
    print("\n[4/5] Testing ResearchTracker YAML & Markdown Sync...")
    test_yaml = "test_research.yaml"
    test_md = "test_changelog.md"
    if os.path.exists(test_yaml): os.remove(test_yaml)
    if os.path.exists(test_md): os.remove(test_md)

    tracker = ResearchTracker(yaml_path=test_yaml, markdown_path=test_md)
    tracker.log_experiment(
        experiment_id="TEST-EXP-001",
        changes_description="Verified memory-efficient Sinkhorn belief head",
        metrics={"loss_policy": 0.01, "loss_value": 0.005, "loss_sinkhorn": 1.9, "loss_entropy": 2.8},
        speed_sps=120.0,
        winrate_vs_heuristic=0.62,
        decision_quality="High aggression with Spies",
        bugs_fixed="None",
        limitations="None",
    )

    assert os.path.isfile(test_yaml), "YAML file should exist"
    assert os.path.isfile(test_md), "Markdown changelog should exist"
    with open(test_md, "r", encoding="utf-8") as f:
        md_text = f.read()
    assert "TEST-EXP-001" in md_text, "Markdown must include logged experiment"
    assert "AUTOMATIC RESEARCH PROTOCOL" in md_text, "Markdown must include standardized protocol header"

    os.remove(test_yaml)
    os.remove(test_md)
    print("  -> ResearchTracker schema and Markdown sync verified!")


def test_fastapi_endpoints():
    print("\n[5/5] Testing FastAPI Human vs AI Endpoints...")
    client = TestClient(app)

    # 1. Test HTML UI Endpoint
    res_ui = client.get("/")
    assert res_ui.status_code == 200
    assert "Salpakan: Game of the Generals" in res_ui.text

    # 2. Test Game State Endpoint
    res_state = client.get("/api/game_state")
    assert res_state.status_code == 200
    state_data = res_state.json()
    assert "pieces" in state_data
    assert "owners" in state_data
    assert "legal_moves" in state_data
    assert "ai_beliefs" in state_data

    # 3. Test New Game Reset Endpoint
    res_new = client.post("/api/new_game")
    assert res_new.status_code == 200

    # 4. Test Player Move Endpoint with first legal move
    legal_map = res_new.json()["legal_moves"]
    assert len(legal_map) > 0, "Human player must have legal moves at start"

    first_from = int(list(legal_map.keys())[0])
    first_to = int(legal_map[str(first_from)][0])

    res_move = client.post("/api/move", json={"from_sq": first_from, "to_sq": first_to})
    assert res_move.status_code == 200
    move_data = res_move.json()
    assert "step_count" in move_data
    assert move_data["step_count"] >= 1

    print("  -> FastAPI Human vs AI endpoints verified!")


if __name__ == "__main__":
    test_checkpoint_manager()
    test_pitfall_detector()
    test_game_recorder_html()
    test_research_tracker()
    test_fastapi_endpoints()
    print("\n=======================================================")
    print("ALL 5 EXTENSION FEATURE TESTS PASSED SUCCESSFULLY!")
    print("=======================================================")
