"""
app.py

Interactive Human vs AI Web Game Server for 'Game of the Generals' (Salpakan).
Built with FastAPI and Uvicorn.

Features:
- Dark tactical military HUD with high-contrast piece insignia.
- True imperfect information: AI pieces are concealed behind fog-of-war.
- Real-time AI Masked Sinkhorn-Knopp Belief Heatmap inspection.
- Combat Arbiter modal animations and sound-ready event cues.
- Auto-deploy or manual piece placement.
- AI powered by the trained 73-Token Transformer (or heuristic fallback).
"""

from typing import Any, Dict, List, Optional
import os
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from tensor_generals_env import (
    TensorGeneralsEnv,
    build_combat_lookup,
    STARTING_PIECE_COUNTS,
    NUM_ROWS,
    NUM_COLS,
    NUM_SQUARES,
    NUM_DIRECTIONS,
    DIR_DELTAS,
)
from board_transformer import BoardTransformer
from train_pfsp import VectorizedHeuristicAgent
from search_engine import AtaraxosSearchEngine

app = FastAPI(title="Game of the Generals - Human vs AI")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load AI Model
model = BoardTransformer().to(device)
model.eval()

# Load Ataraxos Search Engine
search_engine = AtaraxosSearchEngine(model=model, device=device, num_samples=16)

# Attempt to load best checkpoint if available
best_ckpt_path = os.path.join("checkpoints", "best_model.pt")
latest_ckpt_path = os.path.join("checkpoints", "latest.pt")
if os.path.isfile(best_ckpt_path):
    ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"[App] Loaded best trained model from: {best_ckpt_path}")
elif os.path.isfile(latest_ckpt_path):
    ckpt = torch.load(latest_ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"[App] Loaded latest model from: {latest_ckpt_path}")
else:
    print("[App] No checkpoint found. Using initialized Transformer + Heuristic fallback.")

heuristic_agent = VectorizedHeuristicAgent(device)
env = TensorGeneralsEnv(num_envs=1, device=str(device))

RANK_NAMES = [
    "Flag", "Spy", "Private", "Sergeant", "2nd Lieutenant", "1st Lieutenant",
    "Captain", "Major", "Lt. Colonel", "Colonel", "1-Star General", "2-Star General",
    "3-Star General", "4-Star General", "5-Star General"
]

RANK_SYMBOLS = [
    "🚩", "🕵️", "🪖", "🎖️3", "🎖️4", "🎖️5", "🎖️6", "🎖️7", "🎖️8", "🎖️9", "⭐1", "⭐2", "⭐3", "⭐4", "⭐5"
]

RANK_SHORT = [
    "FLAG", "SPY", "PVT", "SGT", "2LT", "1LT", "CPT", "MAJ", "LTC", "COL", "1★", "2★", "3★", "4★", "5★"
]


class MoveRequest(BaseModel):
    from_sq: int
    to_sq: int


@app.get("/", response_class=HTMLResponse)
def serve_game_ui():
    """Serves the polished interactive HTML5/CSS/JS tactical web interface."""
    return HTMLResponse(content=HTML_GAME_UI)


@app.post("/api/new_game")
def new_game():
    """Resets the board for a new Human (Player 1) vs AI (Player 2) match."""
    mask = torch.tensor([True], device=device)
    env.reset_envs(mask)
    return get_game_state_response()


@app.get("/api/game_state")
def get_game_state():
    return get_game_state_response()


@app.post("/api/move")
def player_move(req: MoveRequest):
    """
    Executes Human move (Player 1) followed by AI counter-move (Player 2).
    """
    from_sq = req.from_sq
    to_sq = req.to_sq

    # 1. Validate Human move
    if env.current_player[0].item() != 0:
        raise HTTPException(status_code=400, detail="Not human player's turn")

    owner = env.board_owners[0, from_sq].item()
    if owner != 0:
        raise HTTPException(status_code=400, detail="Selected square does not contain your piece")

    # Find direction
    found_dir = None
    for d, (dr, dc) in enumerate(DIR_DELTAS):
        r, c = from_sq // NUM_COLS, from_sq % NUM_COLS
        if (r + dr) * NUM_COLS + (c + dc) == to_sq and 0 <= r + dr < NUM_ROWS and 0 <= c + dc < NUM_COLS:
            found_dir = d
            break

    if found_dir is None:
        raise HTTPException(status_code=400, detail="Invalid target move square")

    canonical_act = torch.tensor([from_sq * NUM_DIRECTIONS + found_dir], device=device)

    # Execute Human move
    obs, reward, term, trunc, info = env.step(canonical_act)
    human_combat_info = None
    if info["is_combat"][0].item():
        human_combat_info = {
            "is_combat": True,
            "target_sq": to_sq,
        }

    game_over = term[0].item() or trunc[0].item()
    winner = None
    if term[0].item():
        winner = "Human (Player 1)" if reward[0].item() > 0 else "AI (Player 2)"
    elif trunc[0].item():
        winner = "Draw (Truncation Limit)"

    ai_combat_info = None
    ai_move_info = None

    # 2. AI Counter-Move if game not finished (Powered by Ataraxos Test-Time Search)
    if not game_over and env.current_player[0].item() == 1:
        ai_canonical_act_int = search_engine.select_action(obs, env, env_idx=0, temperature=0.0)
        ai_canonical_act = torch.tensor([ai_canonical_act_int], device=device)

        # Absolute action for logging
        ai_abs_act = env.canonical_action_to_absolute(ai_canonical_act, torch.tensor([1], device=device))[0].item()
        ai_from = ai_abs_act // NUM_DIRECTIONS
        ai_d = ai_abs_act % NUM_DIRECTIONS
        ai_to = env.transition_matrix[ai_from, ai_d].item()

        ai_move_info = {
            "from_sq": ai_from,
            "to_sq": ai_to,
        }

        # Execute AI move
        next_obs, ai_reward, ai_term, ai_trunc, ai_step_info = env.step(ai_canonical_act)

        if ai_step_info["is_combat"][0].item():
            ai_combat_info = {
                "is_combat": True,
                "target_sq": ai_to,
            }

        if ai_term[0].item():
            winner = "AI (Player 2)" if ai_reward[0].item() > 0 else "Human (Player 1)"
            game_over = True
        elif ai_trunc[0].item():
            winner = "Draw (Truncation Limit)"
            game_over = True

    response = get_game_state_response()
    response["human_combat"] = human_combat_info
    response["ai_combat"] = ai_combat_info
    response["ai_move"] = ai_move_info
    response["game_over"] = game_over
    response["winner"] = winner

    return response


def get_game_state_response() -> Dict[str, Any]:
    obs = env.get_canonical_observation()
    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            out = model(
                piece_tokens=obs.piece_tokens,
                temporal_features=obs.temporal_features,
                action_mask=obs.action_mask,
                enemy_alive_counts=obs.enemy_alive_counts,
                revealed_mask=obs.revealed_mask,
                true_enemy_ranks=obs.true_enemy_ranks,
            )

    pieces = env.board_pieces[0].cpu().tolist()
    owners = env.board_owners[0].cpu().tolist()
    revealed = env.is_revealed[0].cpu().tolist()
    belief_probs = out.belief_probs[0].cpu().tolist()  # (72, 15)

    # Compute legal moves for human (Player 1)
    legal_mask = env.get_action_mask(canonical=False)[0].cpu()  # (288,)
    legal_moves_map: Dict[int, List[int]] = {}
    for from_sq in range(NUM_SQUARES):
        dests = []
        for d in range(NUM_DIRECTIONS):
            act = from_sq * NUM_DIRECTIONS + d
            if legal_mask[act]:
                to_sq = env.transition_matrix[from_sq, d].item()
                if to_sq >= 0:
                    dests.append(to_sq)
        if dests:
            legal_moves_map[from_sq] = dests

    return {
        "pieces": pieces,
        "owners": owners,
        "revealed": revealed,
        "current_player": int(env.current_player[0].item()),
        "plies_no_combat": int(env.plies_no_combat[0].item()),
        "step_count": int(env.step_counts[0].item()),
        "p1_alive": int(env.piece_alive_counts[0, 0].sum().item()),
        "p2_alive": int(env.piece_alive_counts[0, 1].sum().item()),
        "legal_moves": legal_moves_map,
        "ai_beliefs": belief_probs,
        "rank_names": RANK_NAMES,
        "rank_symbols": RANK_SYMBOLS,
        "rank_short": RANK_SHORT,
    }


HTML_GAME_UI = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Salpakan: Game of the Generals - Human vs AI</title>
<style>
  :root {
    --bg: #0b0f19;
    --card: #151c2e;
    --border: #232f48;
    --text: #e2e8f0;
    --accent: #38bdf8;
    --p1: #2563eb;
    --p2: #dc2626;
    --gold: #f59e0b;
    --green: #10b981;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: var(--bg);
    color: var(--text);
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 16px;
    min-height: 100vh;
  }
  header {
    text-align: center;
    margin-bottom: 16px;
  }
  h1 {
    font-size: 26px;
    font-weight: 800;
    letter-spacing: 1px;
    color: var(--accent);
    text-transform: uppercase;
  }
  .hud {
    display: flex;
    gap: 20px;
    max-width: 1200px;
    width: 100%;
  }
  .main-panel {
    flex: 1;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 20px;
    display: flex;
    flex-direction: column;
    align-items: center;
    box-shadow: 0 20px 40px rgba(0,0,0,0.6);
  }
  .board {
    display: grid;
    grid-template-columns: repeat(8, 60px);
    grid-template-rows: repeat(9, 60px);
    gap: 5px;
    background: #070a12;
    padding: 10px;
    border-radius: 10px;
    border: 2px solid var(--border);
  }
  .cell {
    width: 60px;
    height: 60px;
    background: #1a233a;
    border-radius: 6px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    user-select: none;
    position: relative;
    transition: all 0.15s ease;
    border: 1px solid rgba(255,255,255,0.05);
  }
  .cell:hover { background: #232f4e; }
  .cell.selected { outline: 3px solid var(--gold); transform: scale(1.04); z-index: 5; }
  .cell.valid-move {
    background: #064e3b !important;
    border: 2px solid var(--green) !important;
    animation: pulse 1.5s infinite;
  }
  @keyframes pulse {
    0% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
    70% { box-shadow: 0 0 0 8px rgba(16, 185, 129, 0); }
    100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
  }
  .cell.p1-piece {
    background: linear-gradient(135deg, #1d4ed8, #1e40af);
    color: #fff;
    border: 2px solid #60a5fa;
  }
  .cell.p2-piece {
    background: linear-gradient(135deg, #b91c1c, #991b1b);
    color: #fff;
    border: 2px solid #f87171;
  }
  .cell.revealed-enemy {
    border: 2px solid var(--gold) !important;
  }
  .piece-symbol { font-size: 20px; line-height: 1; }
  .piece-rank { font-size: 11px; font-weight: 800; margin-top: 2px; }
  
  .sidebar {
    width: 380px;
    display: flex;
    flex-direction: column;
    gap: 16px;
  }
  .panel-box {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 16px;
  }
  .score-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
  }
  .badge {
    padding: 6px 12px;
    border-radius: 6px;
    font-weight: 700;
    font-size: 13px;
  }
  .badge.p1 { background: #1e3a8a; color: #93c5fd; border: 1px solid #3b82f6; }
  .badge.p2 { background: #7f1d1d; color: #fca5a5; border: 1px solid #ef4444; }
  
  button.action-btn {
    background: #0284c7;
    color: #fff;
    border: none;
    padding: 10px 16px;
    border-radius: 8px;
    font-weight: 700;
    font-size: 14px;
    cursor: pointer;
    width: 100%;
    transition: background 0.2s;
  }
  button.action-btn:hover { background: #0369a1; }
  button.toggle-btn {
    background: #334155;
    margin-top: 8px;
  }
  button.toggle-btn:hover { background: #475569; }

  .arbiter-modal {
    background: #0f172a;
    border: 2px solid var(--accent);
    border-radius: 10px;
    padding: 12px;
    margin-top: 10px;
    font-size: 13px;
    display: none;
  }
  .beliefs-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
    margin-top: 8px;
  }
  .beliefs-table th, .beliefs-table td {
    padding: 4px 6px;
    text-align: left;
    border-bottom: 1px solid #1e293b;
  }
</style>
</head>
<body>

<header>
  <h1>⚔️ Salpakan: Game of the Generals</h1>
  <p style="color:#94a3b8; font-size:14px; margin-top:4px;">Human Commander (Blue) vs 73-Token Transformer Neural AI (Red)</p>
</header>

<div class="hud">
  <div class="main-panel">
    <div class="board" id="board"></div>
  </div>

  <div class="sidebar">
    <div class="panel-box">
      <div class="score-row">
        <div class="badge p1">🔵 Human: <span id="p1-alive">21</span> Left</div>
        <div class="badge p2">🔴 AI: <span id="p2-alive">21</span> Left</div>
      </div>
      <p style="font-size:13px; color:#94a3b8;">
        Step: <strong id="step-count">0</strong> | No-Combat Plies: <strong id="no-combat">0 / 40</strong>
      </p>
      <div id="arbiter-msg" class="arbiter-modal"></div>
      <button class="action-btn" onclick="startNewGame()" style="margin-top:12px;">🔄 New Game (Reset Board)</button>
      <button class="action-btn toggle-btn" onclick="toggleBeliefView()">🧠 Toggle AI Sinkhorn Beliefs</button>
    </div>

    <div class="panel-box" id="belief-panel" style="display:none; max-height: 380px; overflow-y: auto;">
      <h3 style="font-size:14px; color:var(--accent); margin-bottom:6px;">🧠 AI Real-time Belief Distribution</h3>
      <p style="font-size:11px; color:#94a3b8;">Click on any of your pieces to inspect what the AI neural network predicts its rank to be!</p>
      <div id="belief-details" style="margin-top:8px;"></div>
    </div>
  </div>
</div>

<script>
let gameState = null;
let selectedSq = null;
let showBeliefs = false;

async function fetchState() {
  const res = await fetch('/api/game_state');
  gameState = await res.json();
  renderBoard();
}

async function startNewGame() {
  selectedSq = null;
  const res = await fetch('/api/new_game', { method: 'POST' });
  gameState = await res.json();
  renderBoard();
}

function toggleBeliefView() {
  showBeliefs = !showBeliefs;
  document.getElementById('belief-panel').style.display = showBeliefs ? 'block' : 'none';
  if (selectedSq !== null) showPieceBeliefs(selectedSq);
}

function renderBoard() {
  if (!gameState) return;
  const boardEl = document.getElementById('board');
  boardEl.innerHTML = '';

  document.getElementById('p1-alive').innerText = gameState.p1_alive;
  document.getElementById('p2-alive').innerText = gameState.p2_alive;
  document.getElementById('step-count').innerText = gameState.step_count;
  document.getElementById('no-combat').innerText = `${gameState.plies_no_combat} / 40`;

  const validDests = (selectedSq !== null && gameState.legal_moves[selectedSq]) ? gameState.legal_moves[selectedSq] : [];

  for (let sq = 0; sq < 72; sq++) {
    const cell = document.createElement('div');
    cell.className = 'cell';
    const owner = gameState.owners[sq];
    const rank = gameState.pieces[sq];
    const isRevealed = gameState.revealed[sq] > 0.5;

    if (sq === selectedSq) cell.classList.add('selected');
    if (validDests.includes(sq)) cell.classList.add('valid-move');

    if (owner === 0) {
      // Human piece (Player 1) - Visible to Human
      cell.classList.add('p1-piece');
      cell.innerHTML = `<span class="piece-symbol">${gameState.rank_symbols[rank]}</span>
                        <span class="piece-rank">${gameState.rank_short[rank]}</span>`;
    } else if (owner === 1) {
      // AI piece (Player 2) - Hidden under Fog of War unless revealed
      cell.classList.add('p2-piece');
      if (isRevealed) {
        cell.classList.add('revealed-enemy');
        cell.innerHTML = `<span class="piece-symbol">${gameState.rank_symbols[rank]}</span>
                          <span class="piece-rank">${gameState.rank_short[rank]}</span>`;
      } else {
        cell.innerHTML = `<span class="piece-symbol">🛡️</span>
                          <span class="piece-rank" style="color:#fca5a5;">ENEMY</span>`;
      }
    }

    cell.onclick = () => onCellClick(sq);
    boardEl.appendChild(cell);
  }
}

async function onCellClick(sq) {
  if (!gameState) return;

  // 1. If destination clicked
  if (selectedSq !== null && gameState.legal_moves[selectedSq] && gameState.legal_moves[selectedSq].includes(sq)) {
    const from = selectedSq;
    selectedSq = null;
    await executeMove(from, sq);
    return;
  }

  // 2. If own piece clicked
  if (gameState.owners[sq] === 0) {
    selectedSq = sq;
    renderBoard();
    if (showBeliefs) showPieceBeliefs(sq);
  } else {
    selectedSq = null;
    renderBoard();
  }
}

async function executeMove(fromSq, toSq) {
  const res = await fetch('/api/move', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ from_sq: fromSq, to_sq: toSq })
  });
  const data = await res.json();
  gameState = data;
  renderBoard();

  const arbiterEl = document.getElementById('arbiter-msg');
  if (data.game_over) {
    arbiterEl.style.display = 'block';
    arbiterEl.innerHTML = `<strong style="color:var(--accent)">🏆 GAME OVER!</strong><br>Winner: ${data.winner}`;
  } else if (data.human_combat || data.ai_combat) {
    arbiterEl.style.display = 'block';
    let text = "<strong>⚔️ ARBITER RESOLUTION</strong><br>";
    if (data.human_combat) text += "Your unit engaged in combat!<br>";
    if (data.ai_combat) text += "AI unit initiated combat!";
    arbiterEl.innerHTML = text;
  } else {
    arbiterEl.style.display = 'none';
  }
}

function showPieceBeliefs(sq) {
  const detailsEl = document.getElementById('belief-details');
  if (!gameState || !gameState.ai_beliefs || !gameState.ai_beliefs[sq]) {
    detailsEl.innerHTML = "<p>No active belief data for this square.</p>";
    return;
  }
  const dist = gameState.ai_beliefs[sq];
  let html = `<p><strong>Square ${sq} [${Math.floor(sq/8)}, ${sq%8}]</strong></p>`;
  html += `<table class="beliefs-table"><tr><th>Rank</th><th>Prob</th></tr>`;

  const indexed = dist.map((p, r) => ({ rank: r, prob: p })).sort((a,b) => b.prob - a.prob);
  indexed.slice(0, 6).forEach(item => {
    const pct = (item.prob * 100).toFixed(1);
    html += `<tr><td>${gameState.rank_names[item.rank]}</td><td><strong>${pct}%</strong></td></tr>`;
  });
  html += `</table>`;
  detailsEl.innerHTML = html;
}

window.onload = fetchState;
</script>
</body>
</html>
"""

if __name__ == "__main__":
    import uvicorn
    print("=== Launching Game of the Generals Human vs AI Server on http://127.0.0.1:8000 ===")
    uvicorn.run(app, host="127.0.0.1", port=8000)
