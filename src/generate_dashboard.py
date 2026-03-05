"""
generate_dashboard.py — Generates dashboard.html from real match data.

Called automatically at the end of optimize_params.py, or manually:
    python src/generate_dashboard.py
"""

import json
import os
import sys
from pathlib import Path

import pandas as pd

from config import DEFAULT_CONFIG, WR_ELO, OUTPUT_DIR
from match_predictor import simulate_match, get_tactical_metrics
from squad_builder import build_match_squads

# ── Match ground truth (keep in sync with optimize_params.py) ─────────────────
REAL_RESULTS = [
    {"home": "France",   "away": "Ireland",  "home_score": 36, "away_score": 14, "match_file": "France-Ireland.xlsx"},
    {"home": "Italy",    "away": "Scotland", "home_score": 18, "away_score": 15, "match_file": "Italy-Scotland.xlsx"},
    {"home": "England",  "away": "Wales",    "home_score": 48, "away_score":  7, "match_file": "England-Wales.xlsx"},
    {"home": "Scotland", "away": "England",  "home_score": 31, "away_score": 20, "match_file": "Scotland-England.xlsx"},
    {"home": "Wales",    "away": "France",   "home_score": 12, "away_score": 54, "match_file": "Wales-France.xlsx"},
    {"home": "Ireland",  "away": "Italy",    "home_score": 20, "away_score": 13, "match_file": "Ireland-Italy.xlsx"},
]

FLAGS = {
    "France": "🇫🇷", "Ireland": "🇮🇪", "Italy": "🇮🇹",
    "Scotland": "SCO", "England": "ENG", "Wales": "WAL",
}
FLAG_IS_TEXT = {"Scotland", "England", "Wales"}


# ── Data collection ────────────────────────────────────────────────────────────

def collect_match_data() -> tuple[list[dict], dict]:
    """
    Run predictions for all REAL_RESULTS and collect per-team tactical metrics.

    Returns
    -------
    matches : list of dict
        home, away, pred_margin, actual_margin, correct
    team_metrics : dict
        {team: {elo, pack, control, strike}}
    """
    matches = []
    team_metrics = {}

    for match in REAL_RESULTS:
        try:
            df = build_match_squads(match["match_file"])
        except Exception as exc:
            print(f"  Warning: could not load {match['match_file']}: {exc}")
            continue

        pred = simulate_match(match["home"], match["away"], df, DEFAULT_CONFIG)
        if "error" in pred:
            print(f"  Warning: {pred['error']}")
            continue

        actual = match["home_score"] - match["away_score"]
        correct = (actual > 0 and pred["margin"] > 0) or (actual < 0 and pred["margin"] < 0)

        matches.append({
            "home":          match["home"],
            "away":          match["away"],
            "pred_margin":   round(pred["margin"], 1),
            "actual_margin": actual,
            "correct":       correct,
        })

        # Collect tactical metrics per team (use first appearance)
        for country in [match["home"], match["away"]]:
            if country not in team_metrics:
                m = get_tactical_metrics(df, country, DEFAULT_CONFIG)
                if m:
                    starters = df[(df["country"] == country) & (df["starting"] == 1)]

                    def unit_avg(players, shirt_nos=None, exclude=False):
                        if players.empty:
                            return 70.0
                        if shirt_nos is None:
                            return float(players["skill"].mean())
                        mask = players["shirt_number"].isin(shirt_nos)
                        subset = players[~mask] if exclude else players[mask]
                        return float(subset["skill"].mean()) if not subset.empty else float(players["skill"].mean())

                    fwd = starters[starters["position_group"] == "Forwards"]
                    bck = starters[starters["position_group"] == "Backs"]

                    team_metrics[country] = {
                        "elo":     WR_ELO.get(country, 75.0),
                        "pack":    round(float(fwd["skill"].mean()) if not fwd.empty else 70.0, 1),
                        "control": round(unit_avg(bck, [9, 10]), 1),
                        "strike":  round(unit_avg(bck, [9, 10], exclude=True), 1),
                    }

    return matches, team_metrics


def compute_summary(matches: list[dict]) -> dict:
    """Compute headline metrics from match predictions."""
    if not matches:
        return {"accuracy": 0, "rmse": 0, "mae": 0, "total": 0, "correct": 0}

    import math
    errors = [abs(m["pred_margin"] - m["actual_margin"]) for m in matches]
    correct = sum(1 for m in matches if m["correct"])

    return {
        "total":    len(matches),
        "correct":  correct,
        "accuracy": round(correct / len(matches) * 100, 1),
        "rmse":     round(math.sqrt(sum(e**2 for e in errors) / len(errors)), 1),
        "mae":      round(sum(errors) / len(errors), 1),
    }


# ── HTML template ──────────────────────────────────────────────────────────────

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Six Nations Predictor — Dashboard</title>
<link href="https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  :root {
    --bg: #0b0f1a; --panel: #111827; --border: #1e2d40;
    --gold: #f0b429; --gold-dim: #9a7318; --teal: #38bdf8;
    --red: #f87171; --green: #4ade80; --muted: #4b5a6e;
    --text: #e2e8f0; --subtext: #8b9ab0;
  }
  body { background: var(--bg); color: var(--text); font-family: 'DM Sans', sans-serif; width: 1100px; padding: 40px; }

  header { display: flex; align-items: flex-end; justify-content: space-between; border-bottom: 1px solid var(--border); padding-bottom: 24px; margin-bottom: 32px; }
  .header-left { display: flex; flex-direction: column; gap: 4px; }
  .eyebrow { font-family: 'DM Mono', monospace; font-size: 11px; letter-spacing: 0.2em; color: var(--gold); text-transform: uppercase; }
  h1 { font-family: 'Bebas Neue', sans-serif; font-size: 52px; letter-spacing: 0.04em; line-height: 1; }
  h1 span { color: var(--gold); }
  .subtitle { font-size: 13px; color: var(--subtext); font-weight: 300; margin-top: 6px; }

  .kpi-row { display: flex; gap: 24px; align-items: flex-end; }
  .kpi { text-align: right; }
  .kpi-val { font-family: 'Bebas Neue', sans-serif; font-size: 36px; color: var(--gold); line-height: 1; }
  .kpi-label { font-size: 11px; color: var(--subtext); letter-spacing: 0.1em; text-transform: uppercase; margin-top: 2px; }
  .kpi-sep { width: 1px; height: 40px; background: var(--border); }

  .grid { display: grid; grid-template-columns: 1.15fr 1fr; gap: 20px; }
  .panel { background: var(--panel); border: 1px solid var(--border); border-radius: 12px; padding: 24px 28px; }
  .panel-title { font-family: 'DM Mono', monospace; font-size: 10px; letter-spacing: 0.18em; text-transform: uppercase; color: var(--gold); margin-bottom: 20px; display: flex; align-items: center; gap: 8px; }
  .panel-title::after { content: ''; flex: 1; height: 1px; background: var(--border); }

  .match-row { display: grid; grid-template-columns: 110px 1fr 52px; align-items: center; gap: 12px; margin-bottom: 14px; }
  .match-row:last-child { margin-bottom: 0; }
  .match-label { font-size: 11.5px; font-weight: 500; line-height: 1.3; }
  .match-label span { display: block; font-size: 10px; color: var(--subtext); font-weight: 400; margin-top: 1px; }
  .bar-track { position: relative; height: 36px; display: flex; align-items: center; }
  .bar-track::before { content: ''; position: absolute; left: 50%; top: 4px; bottom: 4px; width: 1px; background: var(--muted); opacity: 0.5; }
  .bar-wrap { position: absolute; height: 12px; border-radius: 2px; }
  .bar-predicted { top: 6px; opacity: 0.45; }
  .bar-actual { top: 20px; }
  .result-actual { font-family: 'DM Mono', monospace; font-size: 12px; font-weight: 500; }
  .result-pred { font-family: 'DM Mono', monospace; font-size: 10px; color: var(--subtext); margin-top: 2px; }
  .badge { display: inline-block; width: 7px; height: 7px; border-radius: 50%; margin-left: 4px; vertical-align: middle; }
  .chart-legend { display: flex; gap: 20px; margin-top: 18px; padding-top: 14px; border-top: 1px solid var(--border); }
  .legend-item { display: flex; align-items: center; gap: 6px; font-size: 11px; color: var(--subtext); }
  .legend-dot { width: 10px; height: 6px; border-radius: 1px; }
  .scale-row { display: grid; grid-template-columns: 110px 1fr 52px; gap: 12px; margin-bottom: 8px; }
  .scale-labels { position: relative; height: 16px; font-family: 'DM Mono', monospace; font-size: 9px; color: var(--muted); }

  .metric-tabs { display: flex; gap: 6px; margin-bottom: 20px; }
  .metric-tab { font-size: 10px; font-family: 'DM Mono', monospace; letter-spacing: 0.08em; padding: 4px 10px; border-radius: 4px; border: 1px solid var(--border); color: var(--subtext); cursor: pointer; transition: all 0.2s; text-transform: uppercase; }
  .metric-tab.active { background: var(--gold); color: var(--bg); border-color: var(--gold); font-weight: 600; }
  .team-row { display: grid; grid-template-columns: 76px 1fr 40px; align-items: center; gap: 12px; margin-bottom: 16px; }
  .team-row:last-child { margin-bottom: 0; }
  .team-name { font-size: 12px; font-weight: 600; }
  .strength-track { height: 8px; background: var(--border); border-radius: 99px; overflow: hidden; }
  .strength-fill { height: 100%; border-radius: 99px; }
  .strength-val { font-family: 'DM Mono', monospace; font-size: 11px; color: var(--subtext); text-align: right; }

  .divider { height: 1px; background: var(--border); margin: 18px 0; }
  .accuracy-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-top: 20px; }
  .acc-cell { background: var(--bg); border: 1px solid var(--border); border-radius: 8px; padding: 14px 12px; text-align: center; }
  .acc-val { font-family: 'Bebas Neue', sans-serif; font-size: 28px; line-height: 1; color: var(--gold); }
  .acc-label { font-size: 9.5px; color: var(--subtext); letter-spacing: 0.1em; text-transform: uppercase; margin-top: 4px; }

  footer { margin-top: 24px; display: flex; justify-content: space-between; align-items: center; padding-top: 16px; border-top: 1px solid var(--border); }
  .footer-note { font-size: 10.5px; color: var(--muted); font-family: 'DM Mono', monospace; }
  .footer-tag { font-size: 10.5px; color: var(--gold-dim); font-family: 'DM Mono', monospace; letter-spacing: 0.1em; }
</style>
</head>
<body>

<header>
  <div class="header-left">
    <div class="eyebrow">Six Nations 2025 · __ROUNDS__ · Hybrid Model</div>
    <h1>MATCH <span>PREDICTOR</span></h1>
    <div class="subtitle">Tactical Squad Analysis × World Rugby ELO Rankings</div>
  </div>
  <div class="kpi-row">
    <div class="kpi">
      <div class="kpi-val">__CORRECT__/__TOTAL__</div>
      <div class="kpi-label">Winners correct</div>
    </div>
    <div class="kpi-sep"></div>
    <div class="kpi">
      <div class="kpi-val">__RMSE__</div>
      <div class="kpi-label">RMSE (pts)</div>
    </div>
    <div class="kpi-sep"></div>
    <div class="kpi">
      <div class="kpi-val">__ACCURACY__%</div>
      <div class="kpi-label">Accuracy</div>
    </div>
  </div>
</header>

<div class="grid">
  <div class="panel">
    <div class="panel-title">Backtesting — Predicted vs Actual Margin</div>
    <div class="scale-row">
      <div></div>
      <div class="scale-labels">
        <span style="position:absolute;left:0">−45</span>
        <span style="position:absolute;left:25%">−22</span>
        <span style="position:absolute;left:50%;transform:translateX(-50%)">0</span>
        <span style="position:absolute;right:25%">+22</span>
        <span style="position:absolute;right:0">+45</span>
      </div>
      <div></div>
    </div>
    <div id="matches"></div>
    <div class="chart-legend">
      <div class="legend-item"><div class="legend-dot" style="background:var(--green);opacity:0.45;"></div>Predicted</div>
      <div class="legend-item"><div class="legend-dot" style="background:var(--green);"></div>Actual (home win)</div>
      <div class="legend-item"><div class="legend-dot" style="background:var(--red);"></div>Actual (away win)</div>
    </div>
  </div>

  <div class="panel">
    <div class="panel-title">Team Strength Ranking</div>
    <div class="metric-tabs" id="metricTabs">
      <div class="metric-tab active" onclick="setMetric('elo')">ELO</div>
      <div class="metric-tab" onclick="setMetric('pack')">Pack</div>
      <div class="metric-tab" onclick="setMetric('control')">Control</div>
      <div class="metric-tab" onclick="setMetric('strike')">Strike</div>
    </div>
    <div id="teamBars"></div>
    <div class="divider"></div>
    <div class="panel-title" style="margin-bottom:0;">Model Performance</div>
    <div class="accuracy-grid">
      <div class="acc-cell"><div class="acc-val">__ACCURACY__%</div><div class="acc-label">Winner acc.</div></div>
      <div class="acc-cell"><div class="acc-val">__RMSE__</div><div class="acc-label">RMSE pts</div></div>
      <div class="acc-cell"><div class="acc-val">__MAE__</div><div class="acc-label">MAE pts</div></div>
      <div class="acc-cell"><div class="acc-val">__TAC_W__/__ELO_W__</div><div class="acc-label">Tac / ELO</div></div>
      <div class="acc-cell"><div class="acc-val">+__HOME_ADV__</div><div class="acc-label">Home adv.</div></div>
      <div class="acc-cell"><div class="acc-val">__DB_SIZE__</div><div class="acc-label">Players in DB</div></div>
    </div>
  </div>
</div>

<footer>
  <div class="footer-note">Model: Hybrid Tactical + ELO · Optimised via grid search · github.com/IvnVillar/six-nations-predictor</div>
  <div class="footer-tag">Six Nations 2025</div>
</footer>

<script>
const MATCHES      = __MATCHES_JSON__;
const TEAM_METRICS = __METRICS_JSON__;
const FLAGS        = __FLAGS_JSON__;
const FLAG_IS_TEXT = __FLAG_IS_TEXT_JSON__;
const MAX_MARGIN   = 45;
const COLORS       = ['#f0b429','#38bdf8','#818cf8','#4ade80','#f97316','#f87171'];

function pct(v) { return Math.min(Math.abs(v) / MAX_MARGIN * 50, 50); }

// Render backtesting chart
const mc = document.getElementById('matches');
MATCHES.forEach(m => {
  const ac = m.actual_margin > 0 ? 'var(--green)' : 'var(--red)';
  const pc = m.pred_margin   > 0 ? 'var(--green)' : 'var(--red)';
  const aStyle = m.actual_margin > 0 ? `left:50%;width:${pct(m.actual_margin)}%` : `right:50%;width:${pct(m.actual_margin)}%`;
  const pStyle = m.pred_margin   > 0 ? `left:50%;width:${pct(m.pred_margin)}%`   : `right:50%;width:${pct(m.pred_margin)}%`;
  const sign = v => v > 0 ? '+'+v : ''+v;
  const badgeColor = m.correct ? 'var(--green)' : 'var(--red)';
  const fmt = t => FLAG_IS_TEXT[t]
    ? `<span style="font-size:8px;font-family:'DM Mono',monospace;font-weight:600;background:var(--border);color:var(--subtext);padding:1px 5px;border-radius:3px;margin-right:3px;">${FLAGS[t]}</span>`
    : FLAGS[t]+' ';
  mc.innerHTML += `
    <div class="match-row">
      <div class="match-label">${fmt(m.home)}${m.home}<span>vs ${fmt(m.away)}${m.away}</span></div>
      <div class="bar-track">
        <div class="bar-wrap bar-predicted" style="${pStyle};background:${pc};"></div>
        <div class="bar-wrap bar-actual"    style="${aStyle};background:${ac};"></div>
      </div>
      <div>
        <div class="result-actual">${sign(m.actual_margin)}<span class="badge" style="background:${badgeColor};"></span></div>
        <div class="result-pred">${sign(m.pred_margin)}</div>
      </div>
    </div>`;
});

// Render team bars
function renderTeams(metric) {
  const data   = Object.entries(TEAM_METRICS).map(([t,v]) => [t, v[metric]]);
  const sorted = data.sort((a,b) => b[1]-a[1]);
  const max = sorted[0][1], min = sorted[sorted.length-1][1], range = max - min;
  document.getElementById('teamBars').innerHTML = sorted.map(([team, val], i) => {
    const fill = range > 0 ? 20 + ((val - min) / range) * 80 : 60;
    const f = FLAG_IS_TEXT[team]
      ? `<span style="font-size:8px;font-family:'DM Mono',monospace;font-weight:600;background:var(--border);color:var(--subtext);padding:1px 5px;border-radius:3px;margin-right:4px;">${FLAGS[team]}</span>`
      : FLAGS[team]+' ';
    return `<div class="team-row">
      <div class="team-name">${f}${team}</div>
      <div class="strength-track"><div class="strength-fill" style="width:${fill}%;background:${COLORS[i]};"></div></div>
      <div class="strength-val">${val.toFixed(1)}</div>
    </div>`;
  }).join('');
}

function setMetric(metric) {
  document.querySelectorAll('.metric-tab').forEach(t =>
    t.classList.toggle('active', t.textContent.toLowerCase() === metric));
  renderTeams(metric);
}

renderTeams('elo');
</script>
</body>
</html>"""


# ── Generator ──────────────────────────────────────────────────────────────────

def generate(output_path: str = None) -> str:
    """
    Build the dashboard HTML and write it to output_path.
    Returns the path of the generated file.
    """
    print("Generating dashboard...")

    matches, team_metrics = collect_match_data()
    summary = compute_summary(matches)

    if not matches:
        raise RuntimeError("No match data available — check that data files are accessible.")

    # Determine rounds label
    n = len(matches)
    rounds_label = f"Rounds 1–{n // 3}" if n >= 3 else f"{n} match{'es' if n > 1 else ''}"

    # Load DB size
    from config import DB_FILE
    try:
        db_size = len(pd.read_csv(DB_FILE))
    except Exception:
        db_size = "—"

    cfg = DEFAULT_CONFIG

    html = HTML_TEMPLATE
    html = html.replace("__ROUNDS__",    rounds_label)
    html = html.replace("__CORRECT__",   str(summary["correct"]))
    html = html.replace("__TOTAL__",     str(summary["total"]))
    html = html.replace("__RMSE__",      str(summary["rmse"]))
    html = html.replace("__MAE__",       str(summary["mae"]))
    html = html.replace("__ACCURACY__",  str(summary["accuracy"]))
    html = html.replace("__TAC_W__",     str(int(cfg["WEIGHT_MODEL_TACTICAL"] * 100)))
    html = html.replace("__ELO_W__",     str(int(cfg["WEIGHT_MODEL_ELO"] * 100)))
    html = html.replace("__HOME_ADV__",  str(cfg["HOME_ADVANTAGE"]))
    html = html.replace("__DB_SIZE__",   str(db_size))
    html = html.replace("__MATCHES_JSON__", json.dumps(matches))
    html = html.replace("__METRICS_JSON__", json.dumps(team_metrics))
    html = html.replace("__FLAGS_JSON__",        json.dumps(FLAGS))
    html = html.replace("__FLAG_IS_TEXT_JSON__", json.dumps({k: True for k in FLAG_IS_TEXT}))

    if output_path is None:
        output_path = os.path.join(OUTPUT_DIR, "dashboard.html")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Dashboard generated → {output_path}")
    return output_path


if __name__ == "__main__":
    try:
        generate()
    except Exception as exc:
        import traceback
        print(f"Error generating dashboard: {exc}")
        traceback.print_exc()
        sys.exit(1)