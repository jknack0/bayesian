"""FastAPI dashboard with WebSocket real-time updates."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

app = FastAPI(title="BayesBot Dashboard")

# Connected WebSocket clients
_ws_clients: list[WebSocket] = []


@app.get("/", response_class=HTMLResponse)
async def index():
    return """
    <!DOCTYPE html>
    <html>
    <head><title>BayesBot Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
      body { font-family: monospace; background: #1a1a2e; color: #e0e0e0; margin: 20px; }
      .panel { background: #16213e; border-radius: 8px; padding: 16px; margin: 10px 0; }
      h1 { color: #0f3460; }
      h2 { color: #53d8fb; font-size: 14px; text-transform: uppercase; }
      .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
      #log { max-height: 300px; overflow-y: auto; font-size: 12px; }
      .regime-mr { color: #2ecc71; } .regime-tr { color: #3498db; } .regime-vol { color: #e74c3c; }
    </style>
    </head>
    <body>
      <h1>BayesBot — Live Monitor</h1>
      <div class="grid">
        <div class="panel"><h2>Regime State</h2><div id="regime">Waiting...</div></div>
        <div class="panel"><h2>Risk / Position</h2><div id="risk">Waiting...</div></div>
      </div>
      <div class="panel"><h2>Equity Curve</h2><div id="equity-chart"></div></div>
      <div class="panel"><h2>Decision Log</h2><div id="log"></div></div>
      <script>
        const ws = new WebSocket(`ws://${location.host}/ws`);
        const equityData = {x: [], y: []};
        ws.onmessage = (e) => {
          const msg = JSON.parse(e.data);
          if (msg.type === 'regime') {
            const r = msg.data;
            document.getElementById('regime').innerHTML =
              `<b>${r.regime_name}</b> (${(r.confidence*100).toFixed(1)}%)<br>` +
              r.state_probabilities.map((p,i) => `State ${i}: ${(p*100).toFixed(1)}%`).join('<br>');
          }
          if (msg.type === 'equity') {
            equityData.x.push(msg.data.bar_index);
            equityData.y.push(msg.data.equity);
            Plotly.react('equity-chart', [{x:equityData.x, y:equityData.y, type:'scatter'}],
              {margin:{t:10,b:30,l:50,r:10}, paper_bgcolor:'#16213e', plot_bgcolor:'#1a1a2e',
               font:{color:'#e0e0e0'}, xaxis:{title:'Bar'}, yaxis:{title:'Equity ($)'}});
          }
          if (msg.type === 'risk') {
            const d = msg.data;
            document.getElementById('risk').innerHTML =
              `Equity: $${d.equity.toFixed(0)}<br>` +
              `CPPI Cushion: $${d.cushion.toFixed(0)}<br>` +
              `Brake: ${d.brake_status}<br>` +
              `Positions: ${d.positions}`;
          }
          if (msg.type === 'log') {
            const el = document.getElementById('log');
            el.innerHTML = `<div>${msg.data.message}</div>` + el.innerHTML;
          }
        };
      </script>
    </body>
    </html>
    """


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    _ws_clients.append(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        _ws_clients.remove(ws)


async def broadcast(event_type: str, data: dict) -> None:
    """Push an event to all connected dashboard clients."""
    msg = json.dumps({"type": event_type, "data": data})
    for ws in list(_ws_clients):
        try:
            await ws.send_text(msg)
        except Exception:
            _ws_clients.remove(ws)


@app.get("/api/health")
async def health():
    return {"status": "ok", "clients": len(_ws_clients)}
