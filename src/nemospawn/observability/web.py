"""Web UI kanban dashboard with Server-Sent Events for real-time updates."""

from __future__ import annotations

import json
import time
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any

from nemospawn.core.config import AGENTS_SUBDIR, TASKS_SUBDIR, PLANS_SUBDIR
from nemospawn.core.state import atomic_read, get_team_dir, list_json_files


def _collect_board_data(team_id: str) -> dict[str, Any]:
    """Collect all board data for SSE streaming."""
    team_dir = get_team_dir(team_id)

    # Tasks
    tasks = []
    for f in list_json_files(team_dir / TASKS_SUBDIR):
        data = atomic_read(f)
        if data:
            tasks.append(data)

    # Agents
    agents = []
    for f in list_json_files(team_dir / AGENTS_SUBDIR):
        data = atomic_read(f)
        if data:
            agents.append(data)

    # Plans
    plans = []
    plans_dir = team_dir / PLANS_SUBDIR
    if plans_dir.is_dir():
        for f in list_json_files(plans_dir):
            data = atomic_read(f)
            if data:
                plans.append(data)

    return {"tasks": tasks, "agents": agents, "plans": plans}


WEB_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NemoSpawn Board — {team_id}</title>
<style>
:root {{
  --bg: #0d1117; --surface: #161b22; --border: #30363d;
  --text: #c9d1d9; --text-dim: #8b949e; --accent: #58a6ff;
  --green: #3fb950; --red: #f85149; --yellow: #d29922; --blue: #58a6ff;
}}
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ background: var(--bg); color: var(--text); font-family: -apple-system, 'Segoe UI', monospace; padding: 16px; }}
h1 {{ color: var(--accent); margin-bottom: 16px; font-size: 1.3em; }}
h2 {{ color: var(--text-dim); font-size: 0.95em; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 1px; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; margin-bottom: 24px; }}
.column {{ background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 12px; min-height: 120px; }}
.column-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; padding-bottom: 6px; border-bottom: 1px solid var(--border); }}
.column-header span {{ font-weight: 600; font-size: 0.85em; }}
.badge {{ background: var(--border); border-radius: 10px; padding: 2px 8px; font-size: 0.75em; }}
.card {{ background: var(--bg); border: 1px solid var(--border); border-radius: 6px; padding: 8px 10px; margin-bottom: 6px; font-size: 0.82em; }}
.card .title {{ font-weight: 600; margin-bottom: 3px; }}
.card .meta {{ color: var(--text-dim); font-size: 0.78em; }}
.pending .column-header span {{ color: var(--text-dim); }}
.blocked .column-header span {{ color: var(--yellow); }}
.running .column-header span {{ color: var(--blue); }}
.done .column-header span {{ color: var(--green); }}
.failed .column-header span {{ color: var(--red); }}
.agents-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 10px; margin-bottom: 24px; }}
.agent-card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 10px 12px; }}
.agent-card .name {{ font-weight: 600; color: var(--accent); }}
.agent-card .detail {{ font-size: 0.8em; color: var(--text-dim); margin-top: 2px; }}
.status-dot {{ display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 4px; }}
.status-dot.running {{ background: var(--green); }}
.status-dot.stopped {{ background: var(--red); }}
.status-dot.idle {{ background: var(--yellow); }}
.status-dot.spawning {{ background: var(--blue); }}
.plans-list {{ margin-bottom: 24px; }}
.plan-card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 10px 12px; margin-bottom: 6px; }}
.plan-card .plan-status {{ font-size: 0.78em; font-weight: 600; }}
.plan-card .plan-status.pending {{ color: var(--yellow); }}
.plan-card .plan-status.approved {{ color: var(--green); }}
.plan-card .plan-status.rejected {{ color: var(--red); }}
.footer {{ color: var(--text-dim); font-size: 0.75em; margin-top: 16px; }}
</style>
</head>
<body>
<h1>NemoSpawn Board — {team_id}</h1>

<h2>Tasks</h2>
<div class="grid" id="tasks-grid"></div>

<h2>Agents</h2>
<div class="agents-grid" id="agents-grid"></div>

<div id="plans-section" style="display:none">
<h2>Plans</h2>
<div class="plans-list" id="plans-list"></div>
</div>

<div class="footer">Auto-updating via SSE &mdash; <span id="last-update"></span></div>

<script>
function renderTasks(tasks) {{
  const grid = document.getElementById('tasks-grid');
  const statuses = ['pending','blocked','running','done','failed'];
  const byStatus = {{}};
  statuses.forEach(s => byStatus[s] = []);
  tasks.forEach(t => {{
    const s = t.status || 'pending';
    if (byStatus[s]) byStatus[s].push(t);
  }});
  grid.innerHTML = statuses.map(s => {{
    const items = byStatus[s];
    const cards = items.map(t => {{
      const agent = t.agent_id ? '<div class="meta">Agent: ' + t.agent_id.substring(0,12) + '</div>' : '';
      const vl = t.metadata && t.metadata.val_loss ? '<div class="meta">val_loss: ' + t.metadata.val_loss + '</div>' : '';
      return '<div class="card"><div class="title">' + (t.title||'').substring(0,40) + '</div>' + agent + vl + '</div>';
    }}).join('');
    return '<div class="column ' + s + '"><div class="column-header"><span>' + s.toUpperCase() + '</span><span class="badge">' + items.length + '</span></div>' + cards + '</div>';
  }}).join('');
}}

function renderAgents(agents) {{
  const grid = document.getElementById('agents-grid');
  grid.innerHTML = agents.map(a => {{
    const lc = a.lifecycle ? a.lifecycle.state : a.status;
    const dotClass = lc === 'idle' ? 'idle' : (a.status || 'stopped');
    return '<div class="agent-card">' +
      '<div class="name"><span class="status-dot ' + dotClass + '"></span>' + (a.name||a.agent_id) + '</div>' +
      '<div class="detail">Role: ' + (a.role||'worker') + '</div>' +
      '<div class="detail">GPUs: ' + JSON.stringify(a.gpu_ids||[]) + '</div>' +
      '<div class="detail">Status: ' + (lc||a.status) + '</div>' +
      '</div>';
  }}).join('');
}}

function renderPlans(plans) {{
  const section = document.getElementById('plans-section');
  const list = document.getElementById('plans-list');
  if (!plans || plans.length === 0) {{ section.style.display = 'none'; return; }}
  section.style.display = 'block';
  list.innerHTML = plans.map(p => {{
    return '<div class="plan-card">' +
      '<div><strong>' + (p.title||'') + '</strong> <span class="plan-status ' + (p.status||'pending') + '">[' + (p.status||'pending') + ']</span></div>' +
      '<div class="detail">Agent: ' + (p.agent_id||'') + '</div>' +
      (p.description ? '<div class="detail">' + p.description.substring(0,100) + '</div>' : '') +
      '</div>';
  }}).join('');
}}

const evtSource = new EventSource('/events');
evtSource.onmessage = function(e) {{
  const data = JSON.parse(e.data);
  renderTasks(data.tasks || []);
  renderAgents(data.agents || []);
  renderPlans(data.plans || []);
  document.getElementById('last-update').textContent = new Date().toLocaleTimeString();
}};
evtSource.onerror = function() {{
  document.getElementById('last-update').textContent = 'connection lost';
}};
</script>
</body>
</html>
"""


def _make_handler(team_id: str):
    """Create an HTTP request handler class bound to a team_id."""

    class BoardHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/" or self.path == "/index.html":
                self._serve_html()
            elif self.path == "/events":
                self._serve_sse()
            elif self.path == "/api/board":
                self._serve_json()
            else:
                self.send_error(404)

        def _serve_html(self):
            html = WEB_HTML.format(team_id=team_id)
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(html.encode())

        def _serve_json(self):
            data = _collect_board_data(team_id)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(data).encode())

        def _serve_sse(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            try:
                while True:
                    data = _collect_board_data(team_id)
                    payload = json.dumps(data, default=str)
                    self.wfile.write(f"data: {payload}\n\n".encode())
                    self.wfile.flush()
                    time.sleep(3)
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass

        def log_message(self, format, *args):
            pass  # suppress default logging

    return BoardHandler


def start_web_board(team_id: str, port: int = 8080) -> HTTPServer:
    """Start the web UI board server in a background thread."""
    handler = _make_handler(team_id)
    server = HTTPServer(("0.0.0.0", port), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server
