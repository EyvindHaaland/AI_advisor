#!/usr/bin/env python3
"""
AI Finansradgiver – Web Server
================================
Kjoer denne filen for a fa tilgang til dashboardet fra telefon.
Gir ogsa mulighet til a starte en ny analyse direkte fra telefonen.

Krav: pip install flask
Start: python server.py
"""

import subprocess
import threading
import time
import sys
import os
from pathlib import Path
from datetime import datetime

try:
    from flask import Flask, send_file, jsonify, Response
except ImportError:
    print("\nInstallerer Flask...")
    subprocess.run([sys.executable, "-m", "pip", "install", "flask", "--quiet"])
    from flask import Flask, send_file, jsonify, Response

BASE_DIR = Path(__file__).parent
DASHBOARD_FILE = BASE_DIR / "dashboard.html"
AGENT_FILE     = BASE_DIR / "agent.py"

app = Flask(__name__)

# ── Shared state ──────────────────────────────────────────────────────────────
state = {
    "running":    False,
    "last_run":   None,
    "last_error": None,
    "log_lines":  [],
}
state_lock = threading.Lock()


def run_agent():
    with state_lock:
        state["running"]    = True
        state["last_error"] = None
        state["log_lines"]  = []

    try:
        process = subprocess.Popen(
            [sys.executable, str(AGENT_FILE)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in process.stdout:
            line = line.rstrip()
            with state_lock:
                state["log_lines"].append(line)
                if len(state["log_lines"]) > 200:
                    state["log_lines"] = state["log_lines"][-200:]

        process.wait()
        if process.returncode != 0:
            with state_lock:
                state["last_error"] = f"Agent avsluttet med kode {process.returncode}"

    except Exception as e:
        with state_lock:
            state["last_error"] = str(e)
    finally:
        with state_lock:
            state["running"]  = False
            state["last_run"] = datetime.now().strftime("%d.%m.%Y %H:%M")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Control panel — works great on phone."""
    with state_lock:
        running    = state["running"]
        last_run   = state["last_run"] or "Aldri"
        last_error = state["last_error"]
        log_lines  = list(state["log_lines"])

    dashboard_exists = DASHBOARD_FILE.exists()

    if running:
        status_color = "#ea580c"
        status_text  = "Analyse kjoerer..."
        btn_html     = '<button disabled style="background:#9ca3af;color:#fff;padding:14px 28px;border:none;border-radius:10px;font-size:16px;cursor:not-allowed">Analyse kjoerer...</button>'
        log_html     = "".join(f'<div class="log-line">{l}</div>' for l in log_lines[-40:])
        log_section  = f'<div class="log-box">{log_html}</div>'
        refresh_meta = '<meta http-equiv="refresh" content="4">'
    else:
        status_color = "#16a34a" if not last_error else "#dc2626"
        status_text  = f"Klar &nbsp;·&nbsp; Sist kjoert: {last_run}"
        if last_error:
            status_text += f'<br><small style="color:#dc2626">Feil: {last_error[:200]}</small>'
        btn_html     = '<a href="/run" class="btn-run">Kjor ny analyse</a>'
        log_section  = ""
        refresh_meta = ""

    dashboard_btn = ""
    if dashboard_exists:
        dashboard_btn = '<a href="/dashboard" class="btn-dash">Apne dashboard</a>'

    return f"""<!DOCTYPE html>
<html lang="no">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="theme-color" content="#1e3a5f">
{refresh_meta}
<title>AI Finansradgiver</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
    background:#f0f4f8;color:#1a2035;min-height:100vh}}
  .header{{background:#1e3a5f;color:#fff;padding:20px;text-align:center}}
  .header h1{{font-size:20px;font-weight:700}}
  .header p{{font-size:13px;opacity:.7;margin-top:4px}}
  .card{{background:#fff;border-radius:14px;padding:20px;margin:16px;
    box-shadow:0 2px 8px rgba(0,0,0,.08)}}
  .status-dot{{display:inline-block;width:10px;height:10px;border-radius:50%;
    background:{status_color};margin-right:8px}}
  .status-text{{font-size:13px;color:#6b7280}}
  .btn-run{{display:block;background:#2563eb;color:#fff;text-align:center;
    padding:16px;border-radius:12px;font-size:17px;font-weight:600;
    text-decoration:none;margin-top:16px}}
  .btn-dash{{display:block;background:#f0f4f8;color:#1e3a5f;text-align:center;
    padding:14px;border-radius:12px;font-size:15px;font-weight:500;
    text-decoration:none;margin-top:10px;border:1.5px solid #d1d5db}}
  .log-box{{background:#0f172a;color:#94a3b8;font-family:monospace;font-size:11px;
    padding:12px;border-radius:10px;max-height:280px;overflow-y:auto;margin-top:12px}}
  .log-line{{padding:1px 0;white-space:pre-wrap;word-break:break-all}}
  .info{{font-size:12px;color:#6b7280;text-align:center;margin:8px 16px 20px}}
</style>
</head>
<body>
<div class="header">
  <h1>AI Finansradgiver</h1>
  <p>Oslo Bors – Swing Trading</p>
</div>

<div class="card">
  <div><span class="status-dot"></span><span class="status-text">{status_text}</span></div>
  {btn_html}
  {dashboard_btn}
  {log_section}
</div>

<p class="info">Analysen henter data for alle aksjer pa Oslo Bors<br>og tar vanligvis 2–4 minutter.</p>
</body>
</html>"""


@app.route("/run")
def trigger_run():
    """Start the agent in a background thread."""
    with state_lock:
        already = state["running"]
    if already:
        return index()
    t = threading.Thread(target=run_agent, daemon=True)
    t.start()
    time.sleep(0.3)   # let the thread flip state["running"] = True
    return index()


@app.route("/dashboard")
def serve_dashboard():
    """Serve the generated dashboard HTML."""
    if DASHBOARD_FILE.exists():
        return send_file(DASHBOARD_FILE)
    return '<html><body><p>Ingen dashboard funnet. Kjor en analyse forst.</p><a href="/">Tilbake</a></body></html>'


@app.route("/status.json")
def status_json():
    with state_lock:
        return jsonify({k: v for k, v in state.items() if k != "log_lines"})


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n AI Finansradgiver – Web Server")
    print(" =" * 28)
    print(f" Dashboard-mappe : {BASE_DIR}")
    print()
    print(" Lokal adresse   : http://localhost:8080")
    print(" (Bruk start_server.bat for Cloudflare-URL til telefon)")
    print()
    print(" Trykk Ctrl+C for a stoppe")
    print()
    app.run(host="0.0.0.0", port=8080, debug=False)
