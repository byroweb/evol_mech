r"""Watch a GA run live in the browser: the fitness curve growing generation by generation, the current
generation's scoreboard (every individual, sorted, coloured by how it was made: elite / mutant / crossover /
random), the elite's ancestry as a bracket back through the generations, and the genes the current best has
moved from the archetype.

    .venv\Scripts\python tools\live_ga.py runs\skater_walk_phase1f            # -> http://localhost:8765
    .venv\Scripts\python tools\live_ga.py runs                               # follows whichever run is being written to

Reads what the GA already writes (generations.jsonl, population/gen_NNNN.npz), so it can be started or stopped at
any time while the run is going; the page polls every 5 s. Works on finished runs too. It also films each
generation's champion on the run's first task (one rollout per generation in one spare process, into <run>/live/)
and plays the latest clip, or any earlier one, on the page; --no-clips turns that off.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evol_mech.bodies import get_body  # noqa: E402
from evol_mech.evolve.ga import KIND_NAMES  # noqa: E402
from tools.run_dashboard import _render_clip  # noqa: E402

ANCESTRY_DEPTH = 8
CLIP_W, CLIP_H, CLIP_FPS = 320, 240, 20


def _resolve(run: Path) -> Path:
    """A run folder, or a folder of runs: then whichever run was written to last."""
    if (run / "config.json").exists():
        return run
    return max((d for d in run.iterdir() if (d / "config.json").exists()),
               key=lambda d: (d / "generations.jsonl").stat().st_mtime if (d / "generations.jsonl").exists() else 0)


def _champion_clips(run_root: Path, stop: threading.Event) -> None:
    """Background: whenever a generation appears whose champion has not been filmed, re-simulate it on the run's
    first task at that generation's own seed and write <run>/live/gen_NNNN.mp4 + .json. One rollout per generation
    in one worker process; the GA keeps its own workers."""
    from multiprocessing import Pool
    with Pool(1) as pool:
        while not stop.is_set():
            try:
                run = _resolve(run_root)
                cfg = json.load(open(run / "config.json"))
                out = run / "live"
                out.mkdir(exist_ok=True)
                todo = [f for f in sorted((run / "population").glob("gen_*.npz")) if not (out / (f.stem + ".json")).exists()]
                if todo:
                    f = todo[-1]                                # newest first; older gaps fill in afterwards
                    z = np.load(f)
                    i = int(np.argmax(z["fitness"]))
                    # an elite carried over is the same genome: point at the clip already made instead of a new file
                    same = [j for j in sorted(out.glob("gen_*.json")) if np.array_equal(np.load(j.with_suffix(".npy")), z["genomes"][i])] if any(out.glob("*.npy")) else []
                    if same:
                        m = json.load(open(same[-1]))
                        m.update({"gen": int(f.stem[4:]), "i": i, "ga_fitness": float(z["fitness"][i]), "clip_gen": m.get("clip_gen", m["gen"])})
                        json.dump(m, open(out / (f.stem + ".json"), "w"))
                        continue
                    np.save(out / (f.stem + ".npy"), z["genomes"][i])
                    task = cfg["tasks"][0]
                    seed = int(z["gen_seed"]) if "gen_seed" in z.files else 0
                    m = pool.apply(_render_clip, ((z["genomes"][i], task, seed, cfg.get("body", "biped"), cfg,
                                                   CLIP_W, CLIP_H, CLIP_FPS, str(out / (f.stem + ".mp4"))),))
                    m.update({"gen": int(f.stem[4:]), "clip_gen": int(f.stem[4:]), "i": i, "ga_fitness": float(z["fitness"][i]), "task": task["name"], "seed": seed})
                    json.dump(m, open(out / (f.stem + ".json"), "w"))
                    continue
            except Exception as e:                              # a file mid-write, a renderer hiccup: try again later
                print(f"clip worker: {e}", file=sys.stderr, flush=True)
            stop.wait(3.0)


def _load(run: Path) -> dict:
    run = _resolve(run)
    cfg = json.load(open(run / "config.json"))
    body = get_body(cfg.get("body", "biped"))
    gens = [json.loads(l) for l in open(run / "generations.jsonl")] if (run / "generations.jsonl").exists() else []
    files = sorted((run / "population").glob("gen_*.npz"))
    pops = {}
    for f in files:
        try:
            z = np.load(f)
            pops[int(f.stem[4:])] = {"fitness": z["fitness"], "kinds": z["kinds"], "parents": z["parents"], "genomes": z["genomes"]}
        except Exception:        # the GA may be mid-write
            pass
    out = {"run": run.name, "population": cfg["population"], "generations": cfg["generations"], "elites": cfg.get("elites", 0),
           "tournament_k": cfg.get("tournament_k"), "curve": [], "scoreboard": None, "ancestry": [], "moved": [], "best": None}
    best_so_far = -np.inf
    for g in gens:
        best_so_far = max(best_so_far, g["best"])
        p20 = float(np.percentile(pops[g["gen"]]["fitness"], 20)) if g["gen"] in pops else None
        out["curve"].append({"gen": g["gen"], "best": g["best"], "best_so_far": best_so_far, "median": g["median"], "p20": p20, "wall_s": g.get("gen_wall_s")})
    if not pops:
        return out
    last = max(pops)
    P = pops[last]
    order = np.argsort(-P["fitness"])
    out["scoreboard"] = {"gen": last, "rows": [{"i": int(i), "fitness": float(P["fitness"][i]), "kind": KIND_NAMES.get(int(P["kinds"][i]), "?"),
                                                "parents": [int(x) for x in P["parents"][i]]} for i in order]}
    # the champion's ancestry: parents are indices into the previous generation
    i, g = int(order[0]), last
    out["ancestry"].append({"gen": g, "i": i, "fitness": float(P["fitness"][i]), "kind": KIND_NAMES.get(int(P["kinds"][i]), "?")})
    while g > 0 and len(out["ancestry"]) < ANCESTRY_DEPTH and (g - 1) in pops:
        par = [int(x) for x in pops[g]["parents"][i] if x >= 0]
        if not par:
            break
        g -= 1
        Q = pops[g]
        node = {"gen": g, "parents": [{"i": j, "fitness": float(Q["fitness"][j]), "kind": KIND_NAMES.get(int(Q["kinds"][j]), "?")} for j in par]}
        out["ancestry"].append(node)
        i = max(par, key=lambda j: Q["fitness"][j])          # follow the fitter parent
    # genes the champion has moved from the archetype
    try:
        center = body.archetype_center(cfg.get("archetype"))
        names = body.gene_names() if hasattr(body, "gene_names") else None
        champ = P["genomes"][order[0]]
        if names is None:
            from evol_mech.skater.spec import decode
            a, b = decode(center), decode(champ)
            moved = [(k, float(a[k]), float(b[k])) for k in b if abs(b[k] - a[k]) > 1e-9]
        else:
            moved = [(names[k], float(center[k]), float(champ[k])) for k in range(len(champ)) if abs(champ[k] - center[k]) > 1e-9]
        moved.sort(key=lambda m: -abs(m[2] - m[1]) / (abs(m[1]) + 1e-9))
        out["moved"] = [{"gene": k, "from": a, "to": b} for k, a, b in moved[:20]]
    except Exception as e:                                  # a body without an archetype decoder
        out["moved_error"] = str(e)
    out["best"] = {"gen": last, "fitness": float(P["fitness"][order[0]])}
    clips = []
    for j in (sorted((run / "live").glob("gen_*.json")) if (run / "live").exists() else []):
        try:
            clips.append(json.load(open(j)))
        except Exception:
            pass
    out["clips"] = clips
    return out


PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>GA live</title>
<style>
body{font:14px/1.4 system-ui,sans-serif;margin:16px;background:#111;color:#ddd}
h1{font-size:18px;margin:0 0 8px}h2{font-size:14px;margin:18px 0 6px;color:#aaa;font-weight:600}
.row{display:flex;gap:24px;flex-wrap:wrap}.card{background:#1b1b1b;border:1px solid #2a2a2a;border-radius:8px;padding:12px}
svg text{fill:#bbb;font-size:11px}
.cells{display:grid;grid-template-columns:repeat(16,1fr);gap:3px;width:620px}
.cell{height:26px;border-radius:3px;font-size:10px;display:flex;align-items:center;justify-content:center;color:#000}
.elite{background:#ffd166}.mutant{background:#8ecae6}.crossover{background:#b5e48c}.random{background:#999}
.legend span{display:inline-block;padding:1px 6px;border-radius:3px;margin-right:6px;color:#000;font-size:11px}
table{border-collapse:collapse}td{padding:1px 8px;font-variant-numeric:tabular-nums}
.br{display:flex;gap:10px;align-items:flex-start}.gen{display:flex;flex-direction:column;gap:4px;min-width:92px}
.node{border-radius:4px;padding:3px 6px;font-size:11px;color:#000;text-align:center}
.arrow{color:#666;align-self:center}
</style></head><body>
<h1 id="title">GA live</h1><div id="status" style="color:#888"></div>
<div class="row">
<div class="card"><h2>fitness by generation (best so far, generation best, p20, median)</h2><svg id="curve" width="620" height="260"></svg></div>
<div class="card"><h2 id="sbtitle">scoreboard</h2>
<div class="legend"><span class="elite">elite</span><span class="mutant">mutant</span><span class="crossover">crossover</span><span class="random">random</span> sorted by fitness, hover for the number</div>
<div class="cells" id="cells"></div></div>
</div>
<div class="row">
<div class="card"><h2 id="cliptitle">the champion, filmed</h2>
<video id="clip" width="320" height="240" autoplay loop muted playsinline style="background:#000;border-radius:4px"></video>
<div id="clipmeta" style="font-size:12px;color:#aaa;max-width:320px"></div><div id="cliplist" style="font-size:11px;margin-top:6px"></div></div>
<div class="card"><h2>the champion's ancestry (fitter parent followed; the yellow box each generation is the one kept)</h2><div class="br" id="br"></div></div>
<div class="card"><h2>genes the champion moved from the archetype (largest relative moves)</h2><table id="moved"></table></div>
</div>
<script>
const f3=x=>x==null?'':x.toFixed(3);
let pinned=null;
function showClip(c){ const v=document.getElementById('clip'); const src=`/clip/${c.clip_gen??c.gen}`;
 if(v.getAttribute('data-src')!==src){ v.src=src; v.setAttribute('data-src',src); }
 document.getElementById('cliptitle').textContent=`generation ${c.gen} champion on ${c.task} (seed ${c.seed}): GA fitness ${f3(c.ga_fitness)}, re-simulated ${f3(c.fitness)}`+(c.clip_gen!=null&&c.clip_gen!==c.gen?` (same genome as gen ${c.clip_gen}, its clip)`:'');
 const keys=['time_upright','stride_m','feet_x_m','swing_clear_m','roll_rms','fan_duty','servo_sat_frac'];
 document.getElementById('clipmeta').textContent=keys.filter(k=>c[k]!=null).map(k=>`${k} ${typeof c[k]=='number'?c[k].toFixed(3):c[k]}`).join('  ·  '); }
function clips(s){ const cs=s.clips||[]; if(!cs.length){ document.getElementById('cliptitle').textContent='the champion, filmed (first clip is rendering)'; return; }
 const c=(pinned!=null && cs.find(x=>x.gen===pinned))||cs[cs.length-1]; showClip(c);
 document.getElementById('cliplist').innerHTML='clips: '+cs.map(x=>`<a href="#" style="color:${x.gen===c.gen?'#ffd166':'#8ecae6'};margin-right:6px" onclick="pinned=${x.gen};return false">${x.gen}</a>`).join('')+` <a href="#" style="color:#aaa" onclick="pinned=null;return false">latest</a>`; }
async function tick(){
 let s; try{ s=await (await fetch('/state.json',{cache:'no-store'})).json(); }catch(e){ document.getElementById('status').textContent='server gone'; return; }
 document.getElementById('title').textContent=`${s.run}: generation ${s.curve.length} of ${s.generations}, population ${s.population}, ${s.elites} elites, tournaments of ${s.tournament_k}`;
 const c=s.curve, last=c[c.length-1];
 document.getElementById('status').textContent=last?`best so far ${f3(last.best_so_far)}   last generation best ${f3(last.best)}  median ${f3(last.median)}  (${(last.wall_s||0).toFixed(0)} s per generation)`:'waiting for the first generation';
 // curve
 const svg=document.getElementById('curve'); const W=620,H=260,L=44,B=24,T=8,R=8; let h='';
 if(c.length){
  const ys=c.flatMap(d=>[d.best,d.median,d.p20??d.median]); const y0=Math.max(-3,Math.min(...ys)), y1=Math.max(...ys)+0.05;
  const X=g=>L+(W-L-R)*g/Math.max(1,s.generations-1), Y=v=>T+(H-T-B)*(1-(Math.max(y0,v)-y0)/(y1-y0));
  for(let v=Math.ceil(y0*4)/4; v<=y1; v+=0.25) h+=`<line x1="${L}" x2="${W-R}" y1="${Y(v)}" y2="${Y(v)}" stroke="#2a2a2a"/><text x="2" y="${Y(v)+4}">${v.toFixed(2)}</text>`;
  for(let g=0; g<s.generations; g+=5) h+=`<text x="${X(g)-4}" y="${H-6}">${g}</text>`;
  const path=(k,col,w)=>`<polyline fill="none" stroke="${col}" stroke-width="${w}" points="${c.filter(d=>d[k]!=null).map(d=>X(d.gen)+','+Y(d[k])).join(' ')}"/>`;
  h+=path('median','#666',1)+path('p20','#8ecae6',1)+path('best','#b5e48c',1)+path('best_so_far','#ffd166',2);
 }
 svg.innerHTML=h;
 // scoreboard
 if(s.scoreboard){ document.getElementById('sbtitle').textContent=`scoreboard, generation ${s.scoreboard.gen} (${s.scoreboard.rows.length} evaluated)`;
  document.getElementById('cells').innerHTML=s.scoreboard.rows.map(r=>`<div class="cell ${r.kind}" title="#${r.i} ${r.kind} fitness ${f3(r.fitness)} parents ${r.parents}">${r.fitness>-9.9?r.fitness.toFixed(2):'x'}</div>`).join(''); }
 // ancestry bracket
 const a=s.ancestry; let b='';
 for(let k=a.length-1;k>=0;k--){ const n=a[k];
  if(n.parents){ b+=`<div class="gen"><div style="color:#888;font-size:11px">gen ${n.gen}</div>`+n.parents.map(p=>`<div class="node ${p.kind}">#${p.i} ${f3(p.fitness)}</div>`).join('')+`</div><div class="arrow">&#8594;</div>`; }
  else b+=`<div class="gen"><div style="color:#888;font-size:11px">gen ${n.gen} champion</div><div class="node ${n.kind}" style="outline:2px solid #fff">#${n.i} ${f3(n.fitness)}</div></div>`; }
 document.getElementById('br').innerHTML=b||'<span style="color:#666">no generation written yet</span>';
 clips(s);
 document.getElementById('moved').innerHTML=s.moved.map(m=>`<tr><td>${m.gene}</td><td>${m.from.toFixed(4)}</td><td>&#8594;</td><td>${m.to.toFixed(4)}</td></tr>`).join('')||`<tr><td style="color:#666">${s.moved_error||'none'}</td></tr>`;
}
tick(); setInterval(tick,5000);
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-clips", action="store_true", help="do not film each generation's champion (saves a core)")
    args = ap.parse_args()
    run = args.run

    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.startswith("/state.json"):
                body, ctype = json.dumps(_load(run)).encode(), "application/json"
            elif self.path.startswith("/clip/"):
                f = _resolve(run) / "live" / f"gen_{int(self.path[6:].split('?')[0]):04d}.mp4"
                body, ctype = (f.read_bytes() if f.exists() else b""), "video/mp4"
            else:
                body, ctype = PAGE.encode(), "text/html; charset=utf-8"
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):          # quiet
            pass

    stop = threading.Event()
    if not args.no_clips:
        threading.Thread(target=_champion_clips, args=(run, stop), daemon=True).start()
    print(f"watching {run} at http://localhost:{args.port}  (Ctrl-C to stop)", flush=True)
    try:
        ThreadingHTTPServer(("127.0.0.1", args.port), H).serve_forever()
    finally:
        stop.set()


if __name__ == "__main__":
    main()
