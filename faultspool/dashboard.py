"""Static HTML dashboard: one self-contained file, no JS, no CDN.

Shows failure trends, the biggest failure clusters, and pass/fail rate per
agent version. Commit it, attach it as a CI artifact, or publish it to Pages.
"""
from __future__ import annotations

import datetime as dt
import html
from collections import Counter, defaultdict
from pathlib import Path

CSS = """
:root{--bg:#f7f7f5;--panel:#fff;--ink:#1d1d1b;--muted:#6b6b66;--line:#e4e3de;
--pass:#2f7d4f;--fail:#c2410c;--diff:#b7791f;--err:#7c3aed;--bar:#3b5bdb}
@media (prefers-color-scheme:dark){:root{--bg:#141413;--panel:#1d1d1b;--ink:#ecebe6;--muted:#9b9a93;
--line:#33332f;--pass:#5bbf85;--fail:#f07a45;--diff:#e0b050;--err:#a78bfa;--bar:#7c93f0}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:14px/1.5 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1080px;margin:0 auto;padding:28px 16px 60px}
h1{font-size:22px;margin:0 0 2px}h2{font-size:15px;margin:0 0 12px}
.sub{color:var(--muted);margin:0 0 24px}
.grid{display:grid;gap:14px;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));margin-bottom:14px}
.two{display:grid;gap:14px;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));margin-bottom:14px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:16px;min-width:0}
.tile .v{font-size:26px;font-weight:650;font-variant-numeric:tabular-nums}.tile .k{color:var(--muted);font-size:12px}
table{width:100%;border-collapse:collapse;font-size:13px}th,td{text-align:left;padding:7px 6px;border-top:1px solid var(--line);vertical-align:top}
th{color:var(--muted);font-weight:500;border-top:0}td.n{text-align:right;font-variant-numeric:tabular-nums}
.pill{display:inline-block;padding:1px 8px;border-radius:99px;font-size:12px;color:#fff}
.passes{background:var(--pass)}.still_fails{background:var(--fail)}.fails_differently{background:var(--diff)}.unreplayable{background:var(--muted)}.error{background:var(--err)}
.empty{color:var(--muted);font-style:italic}.wrap{overflow-x:auto}
svg text{fill:var(--muted);font-size:11px}.legend span{margin-right:12px;color:var(--muted);font-size:12px}
.sw{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:4px}
"""

OUTCOME_COLORS = {"passes": "var(--pass)", "still_fails": "var(--fail)",
                  "fails_differently": "var(--diff)", "unreplayable": "var(--muted)", "error": "var(--err)"}


def _e(v) -> str:
    return html.escape(str(v))


def _stacked_version_bars(rows: list) -> str:
    """rows: [(version, {outcome: n})] -> horizontal stacked bars."""
    if not rows:
        return '<p class="empty">No regression runs yet. Run <code>faultspool run</code>.</p>'
    w, bar_h, gap, label_w = 640, 20, 10, 120
    h = len(rows) * (bar_h + gap)
    out = [f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" aria-label="Outcomes per agent version">']
    for i, (ver, counts) in enumerate(rows):
        y = i * (bar_h + gap)
        total = sum(counts.values()) or 1
        x = label_w
        out.append(f'<text x="0" y="{y + 14}">{_e(ver)}</text>')
        for oc in OUTCOME_COLORS:
            n = counts.get(oc, 0)
            if not n:
                continue
            bw = (w - label_w - 50) * n / total
            out.append(f'<rect x="{x:.1f}" y="{y}" width="{bw:.1f}" height="{bar_h}" rx="3" '
                       f'fill="{OUTCOME_COLORS[oc]}"><title>{oc}: {n}</title></rect>')
            x += bw
        rate = counts.get("passes", 0) / total
        out.append(f'<text x="{x + 6:.1f}" y="{y + 14}">{rate:.0%}</text>')
    out.append("</svg>")
    legend = "".join(f'<span><i class="sw" style="background:{c}"></i>{k.replace("_", " ")}</span>'
                     for k, c in OUTCOME_COLORS.items())
    return f'<div class="legend">{legend}</div>' + "".join(out)


def _daily_bars(days: list) -> str:
    """days: [(date_str, failing, total)] -> vertical bars of failing traces with totals as ghost bars."""
    if not days:
        return '<p class="empty">No traces yet.</p>'
    days = days[-30:]
    w, h, pad = 640, 150, 22
    peak = max(t for _, _, t in days) or 1
    bw = (w - 10) / len(days)
    out = [f'<svg viewBox="0 0 {w} {h + pad}" width="100%" role="img" aria-label="Failing traces per day">']
    for i, (d, fail, total) in enumerate(days):
        x = 5 + i * bw
        th, fh = h * total / peak, h * fail / peak
        out.append(f'<rect x="{x:.1f}" y="{h - th:.1f}" width="{max(bw - 3, 1):.1f}" height="{th:.1f}" '
                   f'rx="2" fill="var(--line)"><title>{d}: {total} traces</title></rect>')
        out.append(f'<rect x="{x:.1f}" y="{h - fh:.1f}" width="{max(bw - 3, 1):.1f}" height="{fh:.1f}" '
                   f'rx="2" fill="var(--fail)"><title>{d}: {fail} failing</title></rect>')
    step = max(1, len(days) // 6)
    for i in range(0, len(days), step):
        out.append(f'<text x="{5 + i * bw:.1f}" y="{h + 15}">{_e(days[i][0][5:])}</text>')
    out.append("</svg>")
    legend = ('<div class="legend"><span><i class="sw" style="background:var(--fail)"></i>failing</span>'
              '<span><i class="sw" style="background:var(--line)"></i>all traces</span></div>')
    return legend + "".join(out)


def collect(store) -> dict:
    db = store.db
    days = defaultdict(lambda: [0, 0])
    failing = {r[0] for r in db.execute("SELECT DISTINCT trace_id FROM failures WHERE severity!='low'")}
    for tid, started in db.execute("SELECT trace_id, started_at FROM traces"):
        d = dt.datetime.fromtimestamp(started or 0, dt.timezone.utc).strftime("%Y-%m-%d")
        days[d][1] += 1
        days[d][0] += tid in failing
    kinds = Counter()
    for kind, n in db.execute("SELECT kind, COUNT(*) FROM failures WHERE severity!='low' GROUP BY kind"):
        kinds[kind] = n
    runs = store.runs()
    by_version: dict = {}
    for r in runs:  # last run per version wins, ordered by first appearance
        by_version[r["agent_version"]] = r
    version_rows = []
    for ver, r in by_version.items():
        counts = Counter(o for (o,) in db.execute("SELECT outcome FROM results WHERE run_id=?", (r["run_id"],)))
        version_rows.append((ver, dict(counts)))
    latest = runs[-1] if runs else None
    latest_results = []
    if latest:
        latest_results = [dict(x) for x in db.execute(
            "SELECT test_id, outcome, regressed, detail FROM results WHERE run_id=? "
            "ORDER BY CASE outcome WHEN 'passes' THEN 1 ELSE 0 END, test_id", (latest["run_id"],))]
    return {"stats": store.stats(), "days": sorted((d, v[0], v[1]) for d, v in days.items()),
            "kinds": kinds.most_common(), "clusters": store.clusters()[:15],
            "versions": version_rows, "latest": latest, "latest_results": latest_results}


def render(data: dict, title: str = "Faultspool") -> str:
    s = data["stats"]
    latest = data["latest"]
    rate = latest["summary"].get("pass_rate") if latest else None
    tiles = [("traces", s["traces"]), ("failing traces", s["failing_traces"]),
             ("regression tests", s["tests"]), ("confirmed", s["tests_confirmed"]),
             ("failure clusters", s["clusters"]),
             ("latest pass rate", "—" if rate is None else f"{rate:.0%}")]
    tiles_html = "".join(f'<div class="card tile"><div class="v">{_e(v)}</div><div class="k">{_e(k)}</div></div>'
                         for k, v in tiles)

    peak = max((n for _, n in data["kinds"]), default=1)
    kinds_html = "".join(
        f'<tr><td>{_e(k)}</td><td style="width:55%"><svg viewBox="0 0 100 8" width="100%" height="10" '
        f'preserveAspectRatio="none"><rect width="{100 * n / peak:.1f}" height="8" rx="2" fill="var(--bar)"/>'
        f'</svg></td><td class="n">{n}</td></tr>' for k, n in data["kinds"]) \
        or '<tr><td class="empty" colspan="3">No failures detected.</td></tr>'

    clusters_html = "".join(
        f'<tr><td class="n">{c["size"]}</td><td>{_e(c["label"])}</td><td><code>{_e(c["representative"])}</code></td></tr>'
        for c in data["clusters"]) or '<tr><td class="empty" colspan="3">Run <code>faultspool cluster</code>.</td></tr>'

    results_html = "".join(
        f'<tr><td><code>{_e(r["test_id"])}</code></td><td><span class="pill {_e(r["outcome"])}">'
        f'{_e(r["outcome"].replace("_", " "))}</span>{" ⚠ regressed" if r["regressed"] else ""}</td>'
        f'<td>{_e(r["detail"])}</td></tr>' for r in data["latest_results"][:50]) \
        or '<tr><td class="empty" colspan="3">No runs yet.</td></tr>'
    latest_label = (f'version <b>{_e(latest["agent_version"])}</b> · '
                    f'{dt.datetime.fromtimestamp(latest["created_at"]).strftime("%Y-%m-%d %H:%M")}') if latest else ""

    generated = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{_e(title)}</title>
<style>{CSS}</style></head><body><main>
<h1>{_e(title)}</h1><p class="sub">Agent failure regression report · generated {generated}</p>
<div class="grid">{tiles_html}</div>
<div class="two">
 <section class="card"><h2>Pass rate by agent version</h2>{_stacked_version_bars(data["versions"])}</section>
 <section class="card"><h2>Failing traces per day</h2>{_daily_bars(data["days"])}</section>
</div>
<div class="two">
 <section class="card"><h2>Failure kinds</h2><table><tr><th>kind</th><th></th><th class="n">count</th></tr>{kinds_html}</table></section>
 <section class="card"><h2>Top clusters</h2><div class="wrap"><table><tr><th class="n">size</th><th>root cause</th><th>representative</th></tr>{clusters_html}</table></div></section>
</div>
<section class="card"><h2>Latest run {latest_label}</h2><div class="wrap"><table>
<tr><th>test</th><th>outcome</th><th>detail</th></tr>{results_html}</table></div></section>
</main></body></html>"""


def build(store, out_path: str = "faultspool-report/index.html", title: str = "Faultspool") -> Path:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(render(collect(store), title), encoding="utf-8")
    return p
