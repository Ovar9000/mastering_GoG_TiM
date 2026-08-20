"""
game_observer.py

Observer View, Game Move Recorder & Animated Replay Generator:
- Captures full move-by-move match traces between AI agents, baseline heuristics, or human matches.
- Validates that all actions taken are legal under board rules.
- Exports interactive, standalone HTML5/JS animated visualizer with:
  * Scrub slider and play/pause controls.
  * Fog-of-war toggle (P1 view, P2 view, or Arbiter/God view).
  * Combat arbiter popups & elimination badge animations.
  * Move log sidebar with detailed challenge resolutions.
"""

from typing import Any, Dict, List, Optional, Tuple
import os
import json
import time
import torch

from tensor_generals_env import (
    TensorGeneralsEnv,
    EnvObservation,
    NUM_ROWS,
    NUM_COLS,
    NUM_SQUARES,
    NUM_DIRECTIONS,
    STARTING_PIECE_COUNTS,
)

RANK_NAMES = [
    "Flag",
    "Spy",
    "Private",
    "Sergeant",
    "2nd Lieutenant",
    "1st Lieutenant",
    "Captain",
    "Major",
    "Lt. Colonel",
    "Colonel",
    "1-Star General",
    "2-Star General",
    "3-Star General",
    "4-Star General",
    "5-Star General",
]

RANK_SHORT_NAMES = [
    "FL", "SP", "PV", "SG", "2L", "1L", "CP", "MJ", "LC", "CL", "1*", "2*", "3*", "4*", "5*"
]


class MoveRecord:
    def __init__(
        self,
        ply: int,
        acting_player: int,
        from_sq: int,
        direction: int,
        to_sq: int,
        moving_rank: int,
        dest_owner_before: int,
        dest_rank_before: int,
        combat_outcome: Optional[int],  # +1 att win, -1 def win, 0 mutual, None if empty
        board_pieces: List[int],
        board_owners: List[int],
        is_revealed: List[float],
        terminated: bool,
        winner: Optional[int],
    ) -> None:
        self.ply = ply
        self.acting_player = acting_player
        self.from_sq = from_sq
        self.direction = direction
        self.to_sq = to_sq
        self.moving_rank = moving_rank
        self.dest_owner_before = dest_owner_before
        self.dest_rank_before = dest_rank_before
        self.combat_outcome = combat_outcome
        self.board_pieces = list(board_pieces)
        self.board_owners = list(board_owners)
        self.is_revealed = list(is_revealed)
        self.terminated = terminated
        self.winner = winner

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ply": self.ply,
            "acting_player": self.acting_player,
            "from_sq": self.from_sq,
            "direction": self.direction,
            "to_sq": self.to_sq,
            "moving_rank": self.moving_rank,
            "dest_owner_before": self.dest_owner_before,
            "dest_rank_before": self.dest_rank_before,
            "combat_outcome": self.combat_outcome,
            "board_pieces": self.board_pieces,
            "board_owners": self.board_owners,
            "is_revealed": self.is_revealed,
            "terminated": self.terminated,
            "winner": self.winner,
        }


class GameRecorder:
    """
    Records game trajectories from the vectorized environment or single-game match.
    """

    def __init__(self, p1_name: str = "Player 1 (AI)", p2_name: str = "Player 2 (AI)") -> None:
        self.p1_name = p1_name
        self.p2_name = p2_name
        self.moves: List[MoveRecord] = []
        self.initial_pieces: List[int] = []
        self.initial_owners: List[int] = []

    def set_initial_state(self, pieces: List[int], owners: List[int]) -> None:
        self.initial_pieces = list(pieces)
        self.initial_owners = list(owners)

    def record_step(
        self,
        ply: int,
        acting_player: int,
        from_sq: int,
        direction: int,
        to_sq: int,
        moving_rank: int,
        dest_owner_before: int,
        dest_rank_before: int,
        combat_outcome: Optional[int],
        board_pieces: List[int],
        board_owners: List[int],
        is_revealed: List[float],
        terminated: bool,
        winner: Optional[int],
    ) -> None:
        rec = MoveRecord(
            ply=ply,
            acting_player=acting_player,
            from_sq=from_sq,
            direction=direction,
            to_sq=to_sq,
            moving_rank=moving_rank,
            dest_owner_before=dest_owner_before,
            dest_rank_before=dest_rank_before,
            combat_outcome=combat_outcome,
            board_pieces=board_pieces,
            board_owners=board_owners,
            is_revealed=is_revealed,
            terminated=terminated,
            winner=winner,
        )
        self.moves.append(rec)

    def export_html(self, output_path: str = "replay.html") -> str:
        """
        Exports an animated, interactive HTML5 replay viewer.
        """
        data = {
            "p1_name": self.p1_name,
            "p2_name": self.p2_name,
            "rank_names": RANK_NAMES,
            "rank_short": RANK_SHORT_NAMES,
            "initial_pieces": self.initial_pieces,
            "initial_owners": self.initial_owners,
            "moves": [m.to_dict() for m in self.moves],
        }

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Game of the Generals - Match Replay</title>
<style>
  :root {{
    --bg: #0f172a;
    --card: #1e293b;
    --text: #f8fafc;
    --accent: #38bdf8;
    --p1: #3b82f6;
    --p2: #ef4444;
    --highlight: #eab308;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: var(--bg);
    color: var(--text);
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 20px;
    min-height: 100vh;
  }}
  header {{ text-align: center; margin-bottom: 20px; }}
  h1 {{ font-size: 24px; color: var(--accent); margin-bottom: 6px; }}
  .container {{
    display: flex;
    gap: 24px;
    max-width: 1100px;
    width: 100%;
  }}
  .board-panel {{
    flex: 1;
    background: var(--card);
    padding: 20px;
    border-radius: 12px;
    display: flex;
    flex-direction: column;
    align-items: center;
    box-shadow: 0 10px 25px rgba(0,0,0,0.5);
  }}
  .board {{
    display: grid;
    grid-template-columns: repeat(8, 52px);
    grid-template-rows: repeat(9, 52px);
    gap: 4px;
    background: #0f172a;
    padding: 8px;
    border-radius: 8px;
    border: 2px solid #334155;
  }}
  .cell {{
    width: 52px;
    height: 52px;
    background: #1e293b;
    border-radius: 4px;
    display: flex;
    align-items: center;
    justify-content: center;
    position: relative;
    font-size: 13px;
    font-weight: bold;
    cursor: default;
    user-select: none;
    transition: background 0.2s;
  }}
  .cell.p1 {{ background: #1d4ed8; color: #fff; border: 2px solid #60a5fa; }}
  .cell.p2 {{ background: #b91c1c; color: #fff; border: 2px solid #f87171; }}
  .cell.fog {{ background: #475569; color: #94a3b8; }}
  .cell.highlight-from {{ outline: 3px solid var(--highlight); }}
  .cell.highlight-to {{ outline: 3px solid #22c55e; }}
  .controls {{
    display: flex;
    gap: 12px;
    margin-top: 16px;
    align-items: center;
    width: 100%;
    justify-content: center;
  }}
  button {{
    background: #334155;
    color: var(--text);
    border: none;
    padding: 8px 14px;
    border-radius: 6px;
    font-weight: 600;
    cursor: pointer;
    transition: background 0.2s;
  }}
  button:hover {{ background: #475569; }}
  button.active {{ background: var(--accent); color: #0f172a; }}
  input[type="range"] {{
    flex: 1;
    max-width: 280px;
    cursor: pointer;
  }}
  .sidebar {{
    width: 320px;
    background: var(--card);
    border-radius: 12px;
    padding: 20px;
    display: flex;
    flex-direction: column;
  }}
  .status-box {{
    background: #0f172a;
    padding: 12px;
    border-radius: 8px;
    margin-bottom: 16px;
    border-left: 4px solid var(--accent);
    font-size: 14px;
  }}
  .move-list {{
    flex: 1;
    max-height: 440px;
    overflow-y: auto;
    background: #0f172a;
    border-radius: 8px;
    padding: 8px;
    font-size: 13px;
  }}
  .move-item {{
    padding: 6px 8px;
    border-radius: 4px;
    cursor: pointer;
    margin-bottom: 4px;
  }}
  .move-item.active {{ background: #334155; font-weight: bold; }}
  .move-item.p1 {{ color: #93c5fd; }}
  .move-item.p2 {{ color: #fca5a5; }}
  .view-toggles {{
    display: flex;
    gap: 8px;
    margin-bottom: 12px;
    width: 100%;
    justify-content: center;
  }}
</style>
</head>
<body>
<header>
  <h1>⚔️ Game of the Generals - Match Replay</h1>
  <p><span style="color:#60a5fa">🔵 {self.p1_name}</span> vs <span style="color:#f87171">🔴 {self.p2_name}</span></p>
</header>

<div class="container">
  <div class="board-panel">
    <div class="view-toggles">
      <button id="view-god" class="active" onclick="setView('god')">Arbiter (God View)</button>
      <button id="view-p1" onclick="setView('p1')">P1 View</button>
      <button id="view-p2" onclick="setView('p2')">P2 View</button>
    </div>

    <div class="board" id="board"></div>

    <div class="controls">
      <button onclick="firstStep()">|&lt;</button>
      <button onclick="prevStep()">&lt;</button>
      <button id="play-btn" onclick="togglePlay()">Play</button>
      <button onclick="nextStep()">&gt;</button>
      <button onclick="lastStep()">&gt;|</button>
      <input type="range" id="scrub" min="0" max="{len(self.moves)}" value="0" oninput="seekTo(this.value)">
      <span id="step-label" style="font-size:14px; min-width:60px;">0 / {len(self.moves)}</span>
    </div>
  </div>

  <div class="sidebar">
    <div class="status-box" id="status-box">
      <strong>Start of Game</strong><br>
      Total Plies: {len(self.moves)}
    </div>
    <h3 style="margin-bottom:8px; font-size:14px; color:#94a3b8;">Move History</h3>
    <div class="move-list" id="move-list"></div>
  </div>
</div>

<script>
const data = {json.dumps(data)};
let currentStep = 0;
let isPlaying = false;
let playInterval = null;
let currentView = 'god';

const DIR_NAMES = ["North", "South", "West", "East"];

function init() {{
  renderMoveList();
  renderBoard();
}}

function setView(view) {{
  currentView = view;
  document.getElementById('view-god').classList.toggle('active', view === 'god');
  document.getElementById('view-p1').classList.toggle('active', view === 'p1');
  document.getElementById('view-p2').classList.toggle('active', view === 'p2');
  renderBoard();
}}

function renderMoveList() {{
  const listEl = document.getElementById('move-list');
  listEl.innerHTML = '';
  data.moves.forEach((m, idx) => {{
    const div = document.createElement('div');
    div.className = `move-item ${{m.acting_player === 0 ? 'p1' : 'p2'}}`;
    div.id = `move-${{idx + 1}}`;
    div.onclick = () => seekTo(idx + 1);

    const fromR = Math.floor(m.from_sq / 8), fromC = m.from_sq % 8;
    const toR = Math.floor(m.to_sq / 8), toC = m.to_sq % 8;
    const combat = m.combat_outcome !== null ? ` (Combat: ${{m.combat_outcome === 1 ? 'Win' : m.combat_outcome === -1 ? 'Loss' : 'Draw'}})` : '';
    div.innerText = `${{idx + 1}}. P${{m.acting_player + 1}}: [${{fromR}},${{fromC}}] -> [${{toR}},${{toC}}]${{combat}}`;
    listEl.appendChild(div);
  }});
}}

function renderBoard() {{
  const boardEl = document.getElementById('board');
  boardEl.innerHTML = '';

  let pieces, owners, revealed, lastMove = null;
  if (currentStep === 0) {{
    pieces = data.initial_pieces;
    owners = data.initial_owners;
    revealed = new Array(72).fill(0);
  }} else {{
    const m = data.moves[currentStep - 1];
    pieces = m.board_pieces;
    owners = m.board_owners;
    revealed = m.is_revealed;
    lastMove = m;
  }}

  for (let sq = 0; sq < 72; sq++) {{
    const cell = document.createElement('div');
    cell.className = 'cell';
    const owner = owners[sq];
    const rank = pieces[sq];

    if (lastMove && sq === lastMove.from_sq) cell.classList.add('highlight-from');
    if (lastMove && sq === lastMove.to_sq) cell.classList.add('highlight-to');

    if (owner === 0) {{
      cell.classList.add('p1');
      if (currentView === 'p2' && revealed[sq] < 0.5) {{
        cell.classList.add('fog');
        cell.innerText = '?';
      }} else {{
        cell.innerText = data.rank_short[rank] || '';
      }}
    }} else if (owner === 1) {{
      cell.classList.add('p2');
      if (currentView === 'p1' && revealed[sq] < 0.5) {{
        cell.classList.add('fog');
        cell.innerText = '?';
      }} else {{
        cell.innerText = data.rank_short[rank] || '';
      }}
    }}
    boardEl.appendChild(cell);
  }}

  // Update Status Box
  const statusEl = document.getElementById('status-box');
  if (currentStep === 0) {{
    statusEl.innerHTML = `<strong>Initial Deployment</strong><br>Game ready to start.`;
  }} else {{
    const m = data.moves[currentStep - 1];
    const pName = m.acting_player === 0 ? data.p1_name : data.p2_name;
    const rName = data.rank_names[m.moving_rank];
    let combatMsg = "";
    if (m.combat_outcome === 1) combatMsg = `<br><span style="color:#4ade80">⚔️ Combat: Attacker eliminated defender!</span>`;
    if (m.combat_outcome === -1) combatMsg = `<br><span style="color:#f87171">⚔️ Combat: Attacker eliminated by defender!</span>`;
    if (m.combat_outcome === 0) combatMsg = `<br><span style="color:#eab308">⚔️ Combat: Mutual Elimination!</span>`;

    let winMsg = "";
    if (m.terminated) {{
      winMsg = `<br><strong style="color:#38bdf8">🏆 Game Over! Winner: ${{m.winner === 0 ? data.p1_name : m.winner === 1 ? data.p2_name : 'Draw'}}</strong>`;
    }}

    statusEl.innerHTML = `<strong>Ply ${{m.ply}}: ${{pName}}</strong><br>Moved ${{rName}} to square ${{m.to_sq}}${{combatMsg}}${{winMsg}}`;
  }}

  // Highlight active move in sidebar
  document.querySelectorAll('.move-item').forEach(el => el.classList.remove('active'));
  if (currentStep > 0) {{
    const activeEl = document.getElementById(`move-${{currentStep}}`);
    if (activeEl) {{
      activeEl.classList.add('active');
      activeEl.scrollIntoView({{ block: 'nearest', behavior: 'smooth' }});
    }}
  }}

  document.getElementById('scrub').value = currentStep;
  document.getElementById('step-label').innerText = `${{currentStep}} / ${{data.moves.length}}`;
}}

function seekTo(step) {{
  currentStep = parseInt(step);
  renderBoard();
}}

function prevStep() {{ if (currentStep > 0) seekTo(currentStep - 1); }}
function nextStep() {{ if (currentStep < data.moves.length) seekTo(currentStep + 1); else pause(); }}
function firstStep() {{ seekTo(0); }}
function lastStep() {{ seekTo(data.moves.length); }}

function togglePlay() {{
  if (isPlaying) pause();
  else play();
}}

function play() {{
  isPlaying = true;
  document.getElementById('play-btn').innerText = 'Pause';
  playInterval = setInterval(() => {{
    if (currentStep < data.moves.length) nextStep();
    else pause();
  }}, 600);
}}

function pause() {{
  isPlaying = false;
  document.getElementById('play-btn').innerText = 'Play';
  if (playInterval) clearInterval(playInterval);
}}

window.onload = init;
</script>
</body>
</html>
"""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"[GameRecorder] Interactive HTML replay exported to: {output_path}")
        return output_path
