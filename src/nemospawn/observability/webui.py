"""Self-contained web UI dashboard for NemoSpawn.

A single-file HTTP server using only Python stdlib that serves a live
dashboard with auto-refreshing agent, task, GPU metric, and message panels.

Usage:
    from nemospawn.observability.webui import start_webui
    server = start_webui("my-team", port=8080)
    server.serve_forever()
"""

from __future__ import annotations

import json
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from nemospawn.core.config import (
    AGENTS_SUBDIR,
    TASKS_SUBDIR,
    INBOX_SUBDIR,
    ARTIFACTS_SUBDIR,
    METRICS_SUBDIR,
)
from nemospawn.core.state import atomic_read, get_team_dir, list_json_files

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _read_all_json(directory: Path) -> list[dict]:
    """Read every .json file in *directory* and return a list of dicts."""
    results: list[dict] = []
    for path in list_json_files(directory):
        data = atomic_read(path)
        if data is not None:
            results.append(data)
    return results


def _load_team_info(team_dir: Path) -> dict[str, Any]:
    """Load the team.json meta file."""
    data = atomic_read(team_dir / "team.json")
    return data if data else {}


def _load_agents(team_dir: Path) -> list[dict]:
    return _read_all_json(team_dir / AGENTS_SUBDIR)


def _load_tasks(team_dir: Path) -> list[dict]:
    return _read_all_json(team_dir / TASKS_SUBDIR)


def _load_inbox(team_dir: Path, agent_id: str) -> list[dict]:
    inbox_dir = team_dir / INBOX_SUBDIR / agent_id
    if not inbox_dir.is_dir():
        return []
    return _read_all_json(inbox_dir)


def _load_all_messages(team_dir: Path) -> list[dict]:
    """Load messages across all agent inboxes, sorted newest first."""
    inbox_root = team_dir / INBOX_SUBDIR
    if not inbox_root.is_dir():
        return []
    messages: list[dict] = []
    for sub in sorted(inbox_root.iterdir()):
        if sub.is_dir():
            messages.extend(_read_all_json(sub))
    messages.sort(key=lambda m: m.get("timestamp", ""), reverse=True)
    return messages


def _load_metrics(team_dir: Path) -> list[dict]:
    return _read_all_json(team_dir / METRICS_SUBDIR)


def _load_artifacts(team_dir: Path) -> list[dict]:
    return _read_all_json(team_dir / ARTIFACTS_SUBDIR)


# ---------------------------------------------------------------------------
# HTML dashboard (inlined CSS + JS)
# ---------------------------------------------------------------------------

_DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NemoSpawn Dashboard</title>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#1a1a2e;--surface:#16213e;--card:#0f3460;
  --nv:#76b900;--nv-dark:#5a8f00;--text:#e0e0e0;--muted:#8892a4;
  --pending:#6c757d;--blocked:#e6a817;--running:#0d6efd;
  --done:#198754;--failed:#dc3545;
}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  background:var(--bg);color:var(--text);line-height:1.5;padding:0}
a{color:var(--nv);text-decoration:none}
header{background:linear-gradient(135deg,#0f3460 0%,#1a1a2e 100%);
  border-bottom:3px solid var(--nv);padding:1rem 2rem;display:flex;
  align-items:center;justify-content:space-between}
header h1{font-size:1.4rem;font-weight:700;color:var(--nv)}
header .meta{font-size:.85rem;color:var(--muted)}
header .status-badge{display:inline-block;padding:.15rem .6rem;
  border-radius:4px;font-size:.75rem;font-weight:600;margin-left:.5rem}
header .status-active{background:var(--done);color:#fff}
header .status-inactive{background:var(--failed);color:#fff}
.container{max-width:1440px;margin:0 auto;padding:1.2rem}
.section-title{font-size:1.05rem;font-weight:600;color:var(--nv);
  margin:1.2rem 0 .6rem;text-transform:uppercase;letter-spacing:.06em}

/* Agent cards */
.agents-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:.8rem}
.agent-card{background:var(--surface);border:1px solid #2a3a5c;border-radius:8px;
  padding:.9rem;transition:border-color .2s}
.agent-card:hover{border-color:var(--nv)}
.agent-card .name{font-weight:600;font-size:.95rem;margin-bottom:.25rem}
.agent-card .role{font-size:.8rem;color:var(--muted);margin-bottom:.4rem}
.agent-card .gpu-tag{display:inline-block;background:#263859;padding:.1rem .45rem;
  border-radius:3px;font-size:.75rem;margin-right:.25rem;margin-bottom:.2rem}
.agent-card .status-dot{display:inline-block;width:8px;height:8px;border-radius:50%;
  margin-right:.35rem;vertical-align:middle}

/* Kanban */
.kanban{display:grid;grid-template-columns:repeat(5,1fr);gap:.7rem}
.kanban-col{background:var(--surface);border-radius:8px;padding:.7rem;min-height:120px}
.kanban-col h3{font-size:.85rem;text-transform:uppercase;letter-spacing:.04em;
  margin-bottom:.5rem;display:flex;align-items:center;gap:.4rem}
.kanban-col h3 .count{background:#263859;padding:.05rem .4rem;border-radius:10px;
  font-size:.7rem;font-weight:400}
.task-card{background:var(--card);border-radius:6px;padding:.6rem;margin-bottom:.5rem;
  border-left:3px solid var(--muted);font-size:.82rem}
.task-card .title{font-weight:600;margin-bottom:.2rem;word-break:break-word}
.task-card .agent-label{color:var(--muted);font-size:.75rem}
.task-card .val-loss{color:var(--nv);font-size:.75rem}
.col-pending .task-card{border-left-color:var(--pending)}
.col-blocked .task-card{border-left-color:var(--blocked)}
.col-running .task-card{border-left-color:var(--running)}
.col-done .task-card{border-left-color:var(--done)}
.col-failed .task-card{border-left-color:var(--failed)}

/* GPU metrics */
.gpu-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:.7rem}
.gpu-bar-wrap{background:var(--surface);border-radius:8px;padding:.7rem}
.gpu-bar-wrap .label{font-size:.8rem;margin-bottom:.3rem;display:flex;justify-content:space-between}
.bar-track{background:#263859;border-radius:4px;height:18px;overflow:hidden}
.bar-fill{height:100%;border-radius:4px;transition:width .4s ease}
.bar-fill.low{background:var(--done)}
.bar-fill.mid{background:var(--blocked)}
.bar-fill.high{background:var(--failed)}

/* Messages */
.messages-list{max-height:320px;overflow-y:auto;background:var(--surface);
  border-radius:8px;padding:.7rem}
.msg-item{padding:.45rem 0;border-bottom:1px solid #2a3a5c;font-size:.82rem}
.msg-item:last-child{border-bottom:none}
.msg-item .from{color:var(--nv);font-weight:600}
.msg-item .ts{color:var(--muted);font-size:.72rem;margin-left:.5rem}
.msg-item .body{margin-top:.15rem;color:var(--text)}

/* Artifacts */
.artifacts-table{width:100%;border-collapse:collapse;font-size:.82rem}
.artifacts-table th{text-align:left;padding:.4rem .6rem;border-bottom:2px solid #2a3a5c;
  color:var(--nv);font-weight:600}
.artifacts-table td{padding:.35rem .6rem;border-bottom:1px solid #1e2d4a}
.artifacts-table tr:hover td{background:var(--surface)}

.empty{color:var(--muted);font-size:.85rem;padding:.6rem 0}
.refresh-indicator{font-size:.72rem;color:var(--muted)}

@media(max-width:900px){
  .kanban{grid-template-columns:repeat(2,1fr)}
}
@media(max-width:600px){
  .kanban{grid-template-columns:1fr}
  .agents-grid{grid-template-columns:1fr}
}
</style>
</head>
<body>

<header>
  <div>
    <h1 id="team-name">NemoSpawn Dashboard</h1>
    <span class="meta" id="team-desc"></span>
  </div>
  <div>
    <span class="meta refresh-indicator" id="updated"></span>
    <span class="status-badge status-active" id="team-status"></span>
  </div>
</header>

<div class="container">
  <!-- Agents -->
  <div class="section-title">Agents</div>
  <div class="agents-grid" id="agents-grid"></div>

  <!-- Task Kanban -->
  <div class="section-title">Tasks</div>
  <div class="kanban" id="kanban">
    <div class="kanban-col col-pending" id="col-pending"><h3>Pending <span class="count">0</span></h3></div>
    <div class="kanban-col col-blocked" id="col-blocked"><h3>Blocked <span class="count">0</span></h3></div>
    <div class="kanban-col col-running" id="col-running"><h3>Running <span class="count">0</span></h3></div>
    <div class="kanban-col col-done"    id="col-done"><h3>Done <span class="count">0</span></h3></div>
    <div class="kanban-col col-failed"  id="col-failed"><h3>Failed <span class="count">0</span></h3></div>
  </div>

  <!-- GPU Metrics -->
  <div class="section-title">GPU Metrics</div>
  <div class="gpu-grid" id="gpu-grid"></div>

  <!-- Messages -->
  <div class="section-title">Recent Messages</div>
  <div class="messages-list" id="messages-list"></div>

  <!-- Artifacts -->
  <div class="section-title">Artifacts</div>
  <div style="overflow-x:auto">
    <table class="artifacts-table" id="artifacts-table">
      <thead><tr><th>Name</th><th>Type</th><th>Path</th><th>Created</th></tr></thead>
      <tbody id="artifacts-body"></tbody>
    </table>
  </div>
</div>

<script>
(function(){
  const STATUS_COLORS = {
    running:"#198754",spawning:"#0d6efd",idle:"#6c757d",
    stopped:"#dc3545",error:"#dc3545",active:"#198754"
  };

  function esc(s){
    if(typeof s!=="string") return "";
    const d=document.createElement("div");d.textContent=s;return d.innerHTML;
  }

  function shortTime(ts){
    if(!ts) return "";
    try{const d=new Date(ts);return d.toLocaleTimeString([],{hour:"2-digit",minute:"2-digit",second:"2-digit"});}
    catch(e){return ts;}
  }

  function barClass(pct){return pct<50?"low":pct<80?"mid":"high";}

  async function fetchJSON(url){
    try{const r=await fetch(url);if(!r.ok) return null;return await r.json();}
    catch(e){return null;}
  }

  async function refresh(){
    const [team,agents,tasks,metrics,messages,artifacts] = await Promise.all([
      fetchJSON("/api/team"),
      fetchJSON("/api/agents"),
      fetchJSON("/api/tasks"),
      fetchJSON("/api/metrics"),
      fetchJSON("/api/messages"),
      fetchJSON("/api/artifacts"),
    ]);

    // Header
    if(team){
      document.getElementById("team-name").textContent=team.name||team.team_id||"NemoSpawn";
      document.getElementById("team-desc").textContent=team.description||"";
      const badge=document.getElementById("team-status");
      badge.textContent=team.status||"active";
      badge.className="status-badge "+(team.status==="active"?"status-active":"status-inactive");
    }
    document.getElementById("updated").textContent="Updated "+new Date().toLocaleTimeString();

    // Agents
    const ag=document.getElementById("agents-grid");
    if(agents&&agents.length){
      ag.innerHTML=agents.map(function(a){
        const color=STATUS_COLORS[a.status]||"#6c757d";
        const gpus=(a.gpu_ids||[]).map(function(g){return '<span class="gpu-tag">GPU '+esc(""+g)+'</span>';}).join("");
        return '<div class="agent-card">'
          +'<div class="name"><span class="status-dot" style="background:'+color+'"></span>'+esc(a.name||a.agent_id)+'</div>'
          +'<div class="role">'+esc(a.role||"worker")+'</div>'
          +'<div>'+gpus+'</div>'
          +'<div style="font-size:.75rem;color:var(--muted);margin-top:.3rem">'+esc(a.status||"unknown")
          +(a.task?' &middot; '+esc(a.task):'')+'</div>'
          +'</div>';
      }).join("");
    } else {
      ag.innerHTML='<div class="empty">No agents registered.</div>';
    }

    // Tasks kanban
    const buckets={pending:[],blocked:[],running:[],done:[],failed:[]};
    if(tasks){tasks.forEach(function(t){
      const s=t.status||"pending";
      if(!buckets[s]) buckets[s]=[];
      buckets[s].push(t);
    });}
    ["pending","blocked","running","done","failed"].forEach(function(s){
      const col=document.getElementById("col-"+s);
      const items=buckets[s]||[];
      const header='<h3>'+s.charAt(0).toUpperCase()+s.slice(1)+' <span class="count">'+items.length+'</span></h3>';
      const cards=items.map(function(t){
        const val=t.metadata&&t.metadata.val_loss;
        const deps=(t.blocked_by||[]).length;
        return '<div class="task-card">'
          +'<div class="title">'+esc(t.title||t.task_id)+'</div>'
          +(t.agent_id?'<div class="agent-label">'+esc(t.agent_id)+'</div>':'')
          +(val!=null?'<div class="val-loss">val_loss: '+Number(val).toFixed(4)+'</div>':'')
          +(deps?'<div class="agent-label">deps: '+deps+'</div>':'')
          +'</div>';
      }).join("");
      col.innerHTML=header+cards;
    });

    // GPU metrics
    const gg=document.getElementById("gpu-grid");
    if(metrics&&metrics.length){
      gg.innerHTML=metrics.map(function(m){
        const util=m.gpu_utilization!=null?m.gpu_utilization:(m.utilization!=null?m.utilization:null);
        const memUsed=m.memory_used_mb||m.mem_used_mb||0;
        const memTotal=m.memory_total_mb||m.mem_total_mb||1;
        const memPct=Math.round(memUsed/memTotal*100)||0;
        const pct=util!=null?Math.round(util):memPct;
        const name=m.gpu_name||m.name||("GPU "+(m.gpu_index!=null?m.gpu_index:(m.index!=null?m.index:"?")));
        return '<div class="gpu-bar-wrap">'
          +'<div class="label"><span>'+esc(name)+'</span><span>'+pct+'%</span></div>'
          +'<div class="bar-track"><div class="bar-fill '+barClass(pct)+'" style="width:'+pct+'%"></div></div>'
          +(memUsed?'<div style="font-size:.7rem;color:var(--muted);margin-top:.2rem">Mem: '+memUsed+'/'+memTotal+' MB</div>':'')
          +'</div>';
      }).join("");
    } else {
      gg.innerHTML='<div class="empty">No GPU metrics available.</div>';
    }

    // Messages
    const ml=document.getElementById("messages-list");
    if(messages&&messages.length){
      ml.innerHTML=messages.slice(0,50).map(function(m){
        return '<div class="msg-item">'
          +'<span class="from">'+esc(m.from_agent||"system")+'</span>'
          +(m.to_agent?' &rarr; '+esc(m.to_agent):'')
          +'<span class="ts">'+shortTime(m.timestamp)+'</span>'
          +'<div class="body">'+esc(m.body||"")+'</div>'
          +'</div>';
      }).join("");
    } else {
      ml.innerHTML='<div class="empty">No messages.</div>';
    }

    // Artifacts
    const ab=document.getElementById("artifacts-body");
    if(artifacts&&artifacts.length){
      ab.innerHTML=artifacts.map(function(a){
        return '<tr>'
          +'<td>'+esc(a.name||a.artifact_id||"")+'</td>'
          +'<td>'+esc(a.type||a.artifact_type||"")+'</td>'
          +'<td>'+esc(a.path||a.location||"")+'</td>'
          +'<td>'+shortTime(a.created_at||"")+'</td>'
          +'</tr>';
      }).join("");
    } else {
      ab.innerHTML='<tr><td colspan="4" class="empty">No artifacts.</td></tr>';
    }
  }

  refresh();
  setInterval(refresh, 5000);
})();
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# HTTP request handler
# ---------------------------------------------------------------------------

class _DashboardHandler(BaseHTTPRequestHandler):
    """Handles dashboard page and JSON API requests."""

    # Attached by the factory; avoids global state.
    team_id: str = ""
    team_dir: Path = Path()

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: D401
        """Route access logs through the module logger."""
        logger.debug(fmt, *args)

    # ---- routing -----------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"

        routes: dict[str, Any] = {
            "/": self._serve_dashboard,
            "/api/team": self._api_team,
            "/api/agents": self._api_agents,
            "/api/tasks": self._api_tasks,
            "/api/metrics": self._api_metrics,
            "/api/messages": self._api_messages,
            "/api/artifacts": self._api_artifacts,
        }

        # Dynamic route: /api/inbox/<agent_id>
        if path.startswith("/api/inbox/"):
            agent_id = path[len("/api/inbox/"):]
            if agent_id:
                self._api_inbox(agent_id)
                return

        handler = routes.get(path)
        if handler:
            handler()
        else:
            self._send_json({"error": "not found"}, status=404)

    # ---- response helpers --------------------------------------------------

    def _send_html(self, html: str, status: int = 200) -> None:
        body = html.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, data: Any, status: int = 200) -> None:
        body = json.dumps(data, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    # ---- page --------------------------------------------------------------

    def _serve_dashboard(self) -> None:
        self._send_html(_DASHBOARD_HTML)

    # ---- API endpoints -----------------------------------------------------

    def _api_team(self) -> None:
        data = _load_team_info(self.team_dir)
        if not data:
            data = {"team_id": self.team_id, "name": self.team_id, "status": "active"}
        self._send_json(data)

    def _api_agents(self) -> None:
        self._send_json(_load_agents(self.team_dir))

    def _api_tasks(self) -> None:
        self._send_json(_load_tasks(self.team_dir))

    def _api_inbox(self, agent_id: str) -> None:
        self._send_json(_load_inbox(self.team_dir, agent_id))

    def _api_metrics(self) -> None:
        self._send_json(_load_metrics(self.team_dir))

    def _api_messages(self) -> None:
        self._send_json(_load_all_messages(self.team_dir))

    def _api_artifacts(self) -> None:
        self._send_json(_load_artifacts(self.team_dir))


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def start_webui(team_id: str, port: int = 8080) -> HTTPServer:
    """Create and return an HTTPServer serving the NemoSpawn dashboard.

    Parameters
    ----------
    team_id:
        The team whose state directory to read from.
    port:
        TCP port to bind (default 8080).

    Returns
    -------
    HTTPServer
        A configured server. Call ``server.serve_forever()`` to run it, or
        use it in a ``with`` block.

    Example
    -------
    >>> server = start_webui("my-team", port=9090)
    >>> server.serve_forever()
    """
    team_dir = get_team_dir(team_id)
    if not team_dir.is_dir():
        logger.warning("Team directory %s does not exist yet; it will be "
                       "created on first write.", team_dir)

    # Build a handler subclass with the team context baked in. This avoids
    # module-level globals and keeps multiple servers independent.
    handler = type(
        "_BoundDashboardHandler",
        (_DashboardHandler,),
        {"team_id": team_id, "team_dir": team_dir},
    )

    server = HTTPServer(("0.0.0.0", port), handler)
    logger.info("NemoSpawn dashboard: http://localhost:%d  (team=%s)", port, team_id)
    return server


# Allow running directly: python -m nemospawn.observability.webui <team_id> [port]
if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    if len(sys.argv) < 2:
        print("Usage: python -m nemospawn.observability.webui <team_id> [port]")
        sys.exit(1)

    _team_id = sys.argv[1]
    _port = int(sys.argv[2]) if len(sys.argv) > 2 else 8080

    _server = start_webui(_team_id, _port)
    try:
        _server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down dashboard server.")
        _server.server_close()
