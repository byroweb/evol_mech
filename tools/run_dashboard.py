r"""Per-generation dashboard of a GA run: pick a generation, watch its best
individual play every task of the run, each clip next to that task's metrics.

    .venv\Scripts\python tools\run_dashboard.py runs\skater_v2                       # -> runs/skater_v2/dashboard/index.html
    .venv\Scripts\python tools\run_dashboard.py runs\skater_v2 --gens 0:40:5 --size 480x360 --workers 12

Open the index.html in a browser (double-click; no server needed). Tasks come
from the run's config.json, so a task added to the config shows up as another
card. Clips are re-simulated with the current code at the generation's own
terrain seed (the GA does not record its rollouts), and the fitness the GA
logged is shown next to the re-simulated one so a code change since the run
is visible. Generations whose best is the same genome (elites carried over)
share one set of clips.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from multiprocessing import Pool
from pathlib import Path

import imageio
import mujoco
import numpy as np

from evol_mech.bodies import get_body

HIDE = {"sim_steps", "wall_s", "steps_per_s"}
LABELS = {
    "fitness": "task fitness", "distance_x": "distance x (m)", "drift_y": "drift y (m)",
    "mean_speed": "mean speed (m/s)", "feet_x_m": "feet progress (m)", "stride_m": "realized stride (m/step)", "servo_accel_rms": "servo cmd accel RMS (rad/s2)", "servo_sat_frac": "servo rate-limited (frac)", "time_upright": "time upright (s)", "fell": "fell",
    "unstable": "unstable", "rms_accel": "RMS hip accel (m/s²)", "rms_lat_err": "RMS lateral error (m)",
    "scrub_m": "tread scrub (m)", "cones_hit": "cones hit", "cones": "cones on course",
    "n_flights": "flights", "max_apex_m": "highest flight apex (m)", "max_flight_m": "longest flight (m)",
    "airborne_s": "time airborne (s)", "fan_energy": "fan energy (norm.)", "motor_energy_j": "motor energy (J)",
    "leg_mass_fraction": "leg mass fraction", "leg_mass_penalty": "leg mass penalty",
    "total_mass": "total mass (kg)", "invalid": "invalid body",
}
ORDER = ["fitness", "time_upright", "fell", "unstable", "distance_x", "feet_x_m", "stride_m", "mean_speed", "drift_y", "rms_lat_err",
         "scrub_m", "cones_hit", "cones", "n_flights", "max_apex_m", "max_flight_m", "airborne_s", "rms_accel", "fan_energy", "motor_energy_j", "leg_mass_fraction", "leg_mass_penalty",
         "total_mass", "invalid"]


# ---- one clip: simulate one task for one genome, write mp4, return metrics -------------

def _render_clip(args):
    genome, task, seed, body_name, cfg, w, h, fps, out_path = args
    out_path = Path(out_path)
    body = get_body(body_name)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.distance, cam.elevation, cam.azimuth = 1.2, -15, 135
    state = {"r": None, "next_t": 0.0, "n": 0}
    writer = imageio.get_writer(str(out_path), fps=fps, codec="libx264", pixelformat="yuv420p",
                                macro_block_size=1, output_params=["-crf", "26", "-movflags", "+faststart"])

    def cb(model, data, t):
        if state["r"] is None:
            state["r"] = mujoco.Renderer(model, h, w)
            cam.trackbodyid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hip")
        if t >= state["next_t"]:
            state["r"].update_scene(data, cam)
            writer.append_data(state["r"].render())
            state["n"] += 1
            state["next_t"] += 1.0 / fps

    try:
        res = body.rollout(genome, task, seed=seed,
                           fans_enabled=cfg.get("fans_enabled", True),
                           fan_energy_cost=cfg.get("fan_energy_cost", 0.2),
                           motor_energy_cost=cfg.get("motor_energy_cost", 0.002), frame_cb=cb)
        metrics = {k: (bool(v) if isinstance(v, (bool, np.bool_)) else float(v))
                   for k, v in res.items() if isinstance(v, (int, float, bool, np.integer, np.floating, np.bool_))}
    except Exception as e:      # invalid body: one blank frame and the error
        metrics = {"invalid": True, "error": str(e)[:200]}
        writer.append_data(np.full((h, w, 3), 60, dtype=np.uint8))
    finally:
        if state["r"] is not None:
            state["r"].close()
        writer.close()
    metrics["clip_s"] = state["n"] / fps
    return metrics


# ---- run data --------------------------------------------------------------------------

def load_run(run: Path, gens_spec: str):
    cfg = json.load(open(run / "config.json"))
    files = sorted((run / "population").glob("gen_*.npz"))
    rows = {}
    log = run / "generations.jsonl"
    if log.exists():
        for line in open(log):
            r = json.loads(line)
            rows[r["gen"]] = r
    gens = []
    for f in files:
        d = np.load(f)
        g = int(f.stem.split("_")[1])
        fit = d["fitness"]
        i = int(np.argmax(fit))
        gens.append({"gen": g, "best_index": i, "genome": d["genomes"][i],
                     "fitness_logged": float(fit[i]), "mean": float(np.mean(fit[np.isfinite(fit)])),
                     "median": float(np.median(fit)), "gen_seed": int(d["gen_seed"]) if "gen_seed" in d.files else g,
                     "wall_s": rows.get(g, {}).get("gen_wall_s")})
    if gens_spec != "all":
        parts = [int(x) if x else None for x in gens_spec.split(":")]
        sel = set(range(len(gens))[slice(*parts)])
        gens = [x for x in gens if x["gen"] in sel]
    return cfg, gens


def fmt_phys(p: dict) -> dict:
    return {k: (round(float(v), 4) if isinstance(v, (int, float, np.floating)) else v) for k, v in p.items()}


# ---- page ------------------------------------------------------------------------------

PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>__TITLE__</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root { --bg:#16181d; --panel:#22252b; --panel2:#2b2f37; --text:#e6e6e6; --dim:#9aa0a8; --gold:#ffc400; --line:#3a3f48; --good:#6fd38a; --bad:#ff6b6b; }
* { box-sizing:border-box } body { margin:0; background:var(--bg); color:var(--text); font:14px/1.4 system-ui, Segoe UI, sans-serif }
header { display:flex; flex-wrap:wrap; gap:14px 28px; align-items:center; padding:12px 18px; border-bottom:1px solid var(--line) }
header h1 { font-size:18px; margin:0 } header .dim { color:var(--dim) }
#controls { display:grid; grid-template-columns: 1fr 360px; gap:16px; padding:12px 18px; align-items:start }
@media (max-width: 900px) { #controls { grid-template-columns: 1fr } }
.panel { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:12px }
.row { display:flex; gap:10px; align-items:center; flex-wrap:wrap }
input[type=range] { flex:1; min-width:180px; accent-color:var(--gold) }
button { background:var(--panel2); color:var(--text); border:1px solid var(--line); border-radius:6px; padding:5px 10px; cursor:pointer }
button:hover { border-color:var(--gold) }
#gen { font-size:22px; font-weight:600; min-width:150px }
canvas#chart { width:100%; height:120px; display:block; margin-top:8px }
#cards { display:grid; grid-template-columns: repeat(auto-fill, minmax(360px, 1fr)); gap:14px; padding:0 18px 18px }
.card { background:var(--panel); border:1px solid var(--line); border-radius:8px; overflow:hidden; display:flex; flex-direction:column }
.card h2 { font-size:15px; margin:0; padding:10px 12px 4px } .card .sub { color:var(--dim); font-size:12px; padding:0 12px 8px }
.card video { width:100%; aspect-ratio:4/3; background:#000; display:block }
table { width:100%; border-collapse:collapse; font-size:13px } td { padding:3px 12px; border-top:1px solid var(--line) }
td:last-child { text-align:right; font-variant-numeric:tabular-nums } td.k { color:var(--dim) }
.good { color:var(--good) } .bad { color:var(--bad) }
#body { padding:0 18px 24px } #body table { font-size:12px } #body td { padding:2px 10px }
#body .grid { display:grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap:0 20px }
.note { color:var(--dim); font-size:12px }
.delta { color:var(--dim); font-size:11px; margin-left:6px }
</style></head><body>
<header><h1>__TITLE__</h1><span class="dim">body <b>__BODY__</b> · population __POP__ · __NGEN__ generations · __NTASK__ tasks</span>
<span class="note">clips re-simulated with the current code at each generation's terrain seed; "logged" = the fitness the GA recorded</span></header>
<div id="controls">
  <div class="panel">
    <div class="row"><button id="prev">◀</button><span id="gen"></span><input id="slider" type="range" min="0" max="0" step="1" value="0"><button id="next">▶</button>
      <button id="replay" title="restart every clip together">⟲ replay all</button>
      <label class="dim"><input id="autoplay" type="checkbox" checked> step through generations when the clips end</label></div>
    <canvas id="chart"></canvas>
    <div class="row note" style="margin-top:6px"><span style="color:var(--gold)">●</span> best (logged) <span style="color:#6aa6ff">●</span> mean <span style="color:#fff">○</span> re-simulated best of the selected generation</div>
  </div>
  <div class="panel" id="summary"></div>
</div>
<div id="cards"></div>
<div id="body"><div class="panel"><h2 style="margin:0 0 6px;font-size:15px">Body and gains of this generation's best</h2><div class="grid" id="phys"></div></div></div>
<script id="data" type="application/json">__DATA__</script>
<script>
const D = JSON.parse(document.getElementById('data').textContent);
const G = D.generations, T = D.tasks, LABELS = D.labels, ORDER = D.order, HIDE = new Set(D.hide);
let cur = G.length - 1;
const $ = id => document.getElementById(id);
const slider = $('slider'); slider.max = G.length - 1; slider.value = cur;
const fmt = v => typeof v === 'boolean' ? (v ? 'yes' : 'no') : (v === null || v === undefined) ? '–' : Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 10 ? v.toFixed(2) : v.toFixed(3);
const cls = (k, v) => (k === 'fell' || k === 'unstable' || k === 'invalid') ? (v ? 'bad' : 'good') : '';
function metricKeys(m) { const ks = Object.keys(m).filter(k => !HIDE.has(k) && k !== 'clip_s' && k !== 'error'); ks.sort((a, b) => (ORDER.indexOf(a) + 1 || 999) - (ORDER.indexOf(b) + 1 || 999)); return ks; }
function build() {
  const cards = $('cards'); cards.innerHTML = '';
  for (const t of T) {
    const c = document.createElement('div'); c.className = 'card'; c.dataset.task = t.name;
    const sub = Object.entries(t).filter(([k]) => k !== 'name').map(([k, v]) => `${k} ${v}`).join(' · ');
    c.innerHTML = `<h2>${t.name}</h2><div class="sub">${sub}</div><video muted playsinline loop controls></video><table></table>`;
    cards.appendChild(c);
  }
}
function show(i, replay = true) {
  cur = Math.max(0, Math.min(G.length - 1, i)); slider.value = cur;
  const g = G[cur], prev = G[cur - 1];
  $('gen').textContent = `generation ${g.gen}`;
  for (const card of document.querySelectorAll('.card')) {
    const t = g.tasks[card.dataset.task], v = card.querySelector('video'), tb = card.querySelector('table');
    if (!t) { tb.innerHTML = '<tr><td class="k">not run for this generation</td></tr>'; continue; }
    if (v.getAttribute('src') !== t.video) { v.src = t.video; } else if (replay) { v.currentTime = 0; }
    v.play().catch(() => {});
    const m = t.metrics, pm = prev && prev.tasks[card.dataset.task] ? prev.tasks[card.dataset.task].metrics : null;
    tb.innerHTML = metricKeys(m).map(k => { const d = pm && typeof m[k] === 'number' && typeof pm[k] === 'number' ? m[k] - pm[k] : null;
      return `<tr><td class="k">${LABELS[k] || k}</td><td class="${cls(k, m[k])}">${fmt(m[k])}${d !== null && Math.abs(d) > 1e-9 ? `<span class="delta">${d > 0 ? '+' : ''}${fmt(d)} vs prev</span>` : ''}</td></tr>`; }).join('')
      + (m.error ? `<tr><td class="k">error</td><td class="bad">${m.error}</td></tr>` : '');
  }
  const s = $('summary');
  s.innerHTML = `<table>
    <tr><td class="k">best individual</td><td>#${g.best_index}</td></tr>
    <tr><td class="k">fitness, logged by the GA</td><td>${fmt(g.fitness_logged)}</td></tr>
    <tr><td class="k">fitness, re-simulated now</td><td>${fmt(g.fitness_resim)}</td></tr>
    <tr><td class="k">population mean / median</td><td>${fmt(g.mean)} / ${fmt(g.median)}</td></tr>
    <tr><td class="k">terrain seed</td><td>${g.gen_seed}</td></tr>
    ${g.wall_s ? `<tr><td class="k">generation wall time</td><td>${fmt(g.wall_s)} s</td></tr>` : ''}
    <tr><td class="k">same best as previous generation</td><td>${prev && prev.clip_key === g.clip_key ? 'yes (elite carried over)' : 'no'}</td></tr></table>`;
  const ph = $('phys'); ph.innerHTML = '';
  const keys = Object.keys(g.phys);
  const cols = 3, per = Math.ceil(keys.length / cols);
  for (let c = 0; c < cols; c++) {
    const tb = document.createElement('table');
    tb.innerHTML = keys.slice(c * per, (c + 1) * per).map(k => { const d = prev && typeof g.phys[k] === 'number' && typeof prev.phys[k] === 'number' ? g.phys[k] - prev.phys[k] : null;
      return `<tr><td class="k">${k}</td><td>${fmt(g.phys[k])}${d !== null && Math.abs(d) > 1e-9 ? `<span class="delta">${d > 0 ? '+' : ''}${fmt(d)}</span>` : ''}</td></tr>`; }).join('');
    ph.appendChild(tb);
  }
  drawChart();
}
function drawChart() {
  const cv = $('chart'), ctx = cv.getContext('2d'); const W = cv.width = cv.clientWidth * devicePixelRatio, H = cv.height = cv.clientHeight * devicePixelRatio;
  ctx.clearRect(0, 0, W, H);
  const best = G.map(g => g.fitness_logged), mean = G.map(g => g.mean), resim = G.map(g => g.fitness_resim);
  const all = best.concat(mean, resim).filter(Number.isFinite); let lo = Math.min(...all), hi = Math.max(...all); if (hi - lo < 1e-9) hi = lo + 1;
  const pad = 8 * devicePixelRatio, x = i => pad + (W - 2 * pad) * (G.length > 1 ? i / (G.length - 1) : 0.5), y = v => H - pad - (H - 2 * pad) * (v - lo) / (hi - lo);
  ctx.strokeStyle = '#3a3f48'; ctx.lineWidth = devicePixelRatio; if (lo < 0 && hi > 0) { ctx.beginPath(); ctx.moveTo(pad, y(0)); ctx.lineTo(W - pad, y(0)); ctx.stroke(); }
  const line = (arr, col) => { ctx.strokeStyle = col; ctx.lineWidth = 1.5 * devicePixelRatio; ctx.beginPath(); arr.forEach((v, i) => { if (!Number.isFinite(v)) return; i === 0 ? ctx.moveTo(x(i), y(v)) : ctx.lineTo(x(i), y(v)); }); ctx.stroke(); };
  line(mean, '#6aa6ff'); line(best, '#ffc400');
  ctx.fillStyle = '#fff'; resim.forEach((v, i) => { if (Number.isFinite(v)) { ctx.beginPath(); ctx.arc(x(i), y(v), 2 * devicePixelRatio, 0, 7); ctx.fill(); } });
  ctx.strokeStyle = '#fff'; ctx.lineWidth = devicePixelRatio; ctx.beginPath(); ctx.moveTo(x(cur), pad); ctx.lineTo(x(cur), H - pad); ctx.stroke();
  ctx.fillStyle = '#9aa0a8'; ctx.font = `${11 * devicePixelRatio}px system-ui`; ctx.fillText(hi.toFixed(2), pad, pad + 10 * devicePixelRatio); ctx.fillText(lo.toFixed(2), pad, H - pad - 2);
}
$('chart').addEventListener('click', e => { const r = e.target.getBoundingClientRect(); show(Math.round((e.clientX - r.left - 8) / (r.width - 16) * (G.length - 1))); });
slider.addEventListener('input', () => show(+slider.value));
$('prev').onclick = () => show(cur - 1); $('next').onclick = () => show(cur + 1);
$('replay').onclick = () => document.querySelectorAll('video').forEach(v => { v.currentTime = 0; v.play().catch(() => {}); });
document.addEventListener('keydown', e => { if (e.key === 'ArrowLeft') show(cur - 1); if (e.key === 'ArrowRight') show(cur + 1); });
build();
// step to the next generation when the longest clip has played once (if enabled)
let timer = null;
function arm() { clearTimeout(timer); if (!$('autoplay').checked) return; const g = G[cur]; const longest = Math.max(...Object.values(g.tasks).map(t => t.metrics.clip_s || 0), 1); timer = setTimeout(() => { if ($('autoplay').checked && cur < G.length - 1) { show(cur + 1); arm(); } }, (longest + 0.5) * 1000); }
$('autoplay').addEventListener('change', arm); slider.addEventListener('input', arm); $('prev').addEventListener('click', arm); $('next').addEventListener('click', arm);
show(cur); arm();
window.addEventListener('resize', drawChart);
</script></body></html>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path)
    ap.add_argument("--gens", default="all", help="python slice over generations, e.g. 0:40:2 (default all)")
    ap.add_argument("--size", default="480x360")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", type=Path, default=None, help="default <run>/dashboard")
    args = ap.parse_args()
    run = args.run
    cfg, gens = load_run(run, args.gens)
    w, h = (int(v) for v in args.size.lower().split("x"))
    out = args.out or run / "dashboard"
    (out / "clips").mkdir(parents=True, exist_ok=True)
    body = get_body(cfg.get("body", "biped"))
    tasks = cfg["tasks"]

    # one set of clips per distinct best genome
    jobs, index = [], {}
    for g in gens:
        key = hashlib.sha1(np.ascontiguousarray(g["genome"]).tobytes()).hexdigest()[:10]
        g["clip_key"] = key
        if key in index:
            continue
        index[key] = {}
        for ti, task in enumerate(tasks):
            path = out / "clips" / f"{key}_{task['name']}.mp4"
            index[key][task["name"]] = path
            jobs.append((g["genome"], task, g["gen_seed"] * 1000 + ti, cfg.get("body", "biped"), cfg, w, h, args.fps, str(path)))
    print(f"{len(gens)} generations, {len(index)} distinct bests, {len(jobs)} clips at {w}x{h}")
    if args.workers > 1 and len(jobs) > 1:
        with Pool(args.workers) as pool:
            metrics = pool.map(_render_clip, jobs, chunksize=1)
    else:
        metrics = [_render_clip(j) for j in jobs]
    by_path = {j[-1]: m for j, m in zip(jobs, metrics)}

    out_gens = []
    for g in gens:
        tdata, resim = {}, 0.0
        for task in tasks:
            path = index[g["clip_key"]][task["name"]]
            m = by_path[str(path)]
            tdata[task["name"]] = {"video": f"clips/{path.name}", "metrics": m}
            resim += float(task.get("weight", 1.0)) * float(m.get("fitness", float("nan")))
        out_gens.append({"gen": g["gen"], "best_index": g["best_index"], "fitness_logged": g["fitness_logged"],
                         "fitness_resim": resim, "mean": g["mean"], "median": g["median"], "gen_seed": g["gen_seed"],
                         "wall_s": g["wall_s"], "clip_key": g["clip_key"], "phys": fmt_phys(body.decode(g["genome"])),
                         "tasks": tdata})
    data = {"run": cfg.get("run_name", run.name), "body": cfg.get("body", "biped"), "tasks": tasks,
            "generations": out_gens, "labels": LABELS, "order": ORDER, "hide": sorted(HIDE)}
    (out / "data.json").write_text(json.dumps(data, indent=1))
    page = (PAGE.replace("__TITLE__", html.escape(f"{data['run']}: best of each generation"))
            .replace("__BODY__", html.escape(data["body"])).replace("__POP__", str(cfg.get("population", "?")))
            .replace("__NGEN__", str(len(out_gens))).replace("__NTASK__", str(len(tasks)))
            .replace("__DATA__", json.dumps(data).replace("</", "<\\/")))
    (out / "index.html").write_text(page, encoding="utf-8")
    print(f"wrote {out / 'index.html'}")


if __name__ == "__main__":
    main()
