# evol_mech

Evolving the geometry and gait of a scaled-down, fan-assisted bipedal mech.

Two simulators, two stages:

1. **Evolution** runs in [MuJoCo](https://mujoco.org) on the CPU, many workers in
   parallel. The mech is a fixed-topology parametric body; the genome is a
   bounded parameter vector covering geometry, gait oscillators and fan
   attitude-controller gains.
2. **Validation** (later) exports elites to URDF and runs them in ROS2 + Gazebo
   under WSL2.

## Layout

    evol_mech/genome   parameter vector, bounds, mutation, crossover
    evol_mech/body     genome -> MJCF (and later URDF)
    evol_mech/sim      terrains, controllers, rollout, fitness tasks
    evol_mech/skater   the stage-1 skater mech: its own genome, body, controller, tasks
    evol_mech/bodies   body registry: config "body": "biped" (default) or "skater"
    evol_mech/evolve   GA orchestrator and worker pool
    evol_mech/data     run directories, result logging, video export
    tools/             benchmark and inspection scripts
    configs/           run configurations
    runs/              experiment outputs (gitignored)

## Quick start

    python -m venv .venv
    .venv\Scripts\pip install -e .[dev]
    .venv\Scripts\python tools\benchmark.py        # steps/sec per worker
    .venv\Scripts\python -m evol_mech.evolve.run configs\default.json
    .venv\Scripts\python tools\view.py runs\dev\best.json   # interactive viewer
    .venv\Scripts\python tools\plot_linkage.py runs\dev\best.json -o leg.png
    .venv\Scripts\python tools\evolution_video.py runs\dev    # every individual per generation, best outlined, children branching
    .venv\Scripts\python tools
un_dashboard.py runs\dev      # browser dashboard: pick a generation, its best plays every task next to that task's metrics

## Skater mech (stage 1)

`docs/skater_design.md` describes the next target body; the build follows
archetype B: three metal-gear servos per leg in the hip (roll, thigh, and shin
through a parallelogram knee, as on Caltech LEO), a passive two-axis sprung ankle,
powered tread feet, and two pitch-only fans, travelling mostly by skating.
Its skeleton is in `evol_mech/skater/` and shares the GA, worker pool,
terrains, store and video with the biped; the genome, body, controller and
tasks are its own. `docs/skater_review.md` maps the design to the code, lists
where the build deliberately departs from it, and the open questions.

    .venv\Scripts\python tools\skater_snapshot.py                 # stills, summary, 8 s skate clip
    .venv\Scripts\python tools\skater_layout.py                   # annotated servo-layout drawing
    tools\mechanism_checker.html                                  # open in a browser: pose the servos, horns and links, see range / singularity / interference checks
    .venv\Scripts\python tools\view.py --body skater --speed 0.8   # interactive; --objective slalom|stand
    .venv\Scripts\python -m evol_mech.evolve.run configs\skater.json
    .venv\Scripts\python -m evol_mech.evolve.run configs\skater.json --lock-groups geometry,leg,foot,drivetrain,fan_spec   # gains only

## Designing an archetype by hand

Open `tools/linkage_editor.html` in a browser (double-click; no server needed).
Drag the linkage points A/B/C/D/P from the design sketch, watch the foot path,
tag the body parts with your own names, and **Export JSON**. Then seed a run
from it:

    .venv\Scripts\python -m evol_mech.evolve.run configs\default.json --archetype path\to\my_archetype.json

or set `"archetype": "configs/archetypes/my_archetype.json"` in the config.
The initial population is sampled around the archetype, which is itself kept
as individual 0 of generation 0. `configs/archetypes/default.json` is the
built-in default body. Values outside the genome bounds are flagged in the
editor and rejected on load.

## Design notes

**Body.** Hip (steppers inside) with two identical crank-rocker 4-bar legs in
the sagittal plane (crank pivot A, crank tip D, rocker pivot B, joint C); the
foot P is a flat pad rigid to the coupler CD, placed a fraction along CD and a
height below it, so the foot path is set by link lengths and that triangle. The
core sits on the hip, rigidly by default or on a ±30° yaw pivot when the run
config sets `"waist_yaw": true` (the yaw gains in the genome are inert
otherwise). It holds a 4S battery, lying flat with its long axis fore/aft, so the
core lengthens with capacity and is never wider than the hip allows, so
narrow hips still clear the legs; capacity and position are in
the genome. The fans are placed by the gap between the prop's inboard edge
and the core side, so they move outward with prop size and core width, and
their height is a fraction of the core height, so they never sit above it. Two pitch-vectored fans sit on the core. The foot
is a gene: a flat pad along the coupler or a ball end (rotation-invariant
contact).

**Locking genes.** The run config's `"locked"` list (or `--lock a,b` on the
command line) holds those genes at the archetype's value through initialisation,
crossover and mutation, e.g. `--lock foot_type` to evolve around a chosen foot.
Foot types: 0 flat pad, 1 ball, 2 cross (pad plus a lateral bar whose half-width
is a gene, capped so the feet never meet). Two point feet on a narrow hip have
no roll stiffness: with antiphase cranks the body rolls onto the lifted foot.
`--lock-groups geometry,linkage,drivetrain,fan_spec` freezes the whole body and
evolves only the gait and fan-controller genes.

**Masses are derived, not evolved** (`body/structure.py`): hip and core shells
weigh what printed boxes of their size weigh, bracket posts and fan arms by
length, plus a 250 g electronics allowance and a battery mass that follows
capacity. Fan and stepper masses come from their specs. Catalogue parts are
snapped on decode: whole-inch props, whole blades, whole-mm stator sizes,
100 mAh battery steps. Linkages that are not rotatable crank-rockers are rejected before
simulation (`body/linkage.py`). Both fixed pivots A and B are mounted on the
hip, never the core, so the waist angle does not change the foot locus. B sits
on a post above the compact hip; `body/clearance.py` rejects any body whose leg
links would touch the yaw-swept core or the swept fan discs.

**Hardware models** (`body/hardware.py`). Fans: thrust from momentum theory
with motor power proportional to stator volume, mass from the same spec,
spin-up as a first-order lag. Steppers: velocity actuator with an angle
tracking term, torque-limited so they stall under overload; mass follows
holding torque and lands in the hip. All constants are documented and meant to
be replaced by bench measurements.

**Controller** (`sim/controller.py`). Steppers track a constant crank speed
(sign = direction) with a right/left phase offset; the stand task parks both
cranks where the feet are level and centred. Fans run a PD attitude loop:
roll to differential throttle, pitch to fan pitch. Gains are in the genome.

**Fitness** (`sim/rollout.py`). Every task spawns in the parked stance with
both feet level. Skate: forward distance scaled by the fraction
of the task spent upright, minus lateral drift,
fall penalty and fan energy. Walk: progress of the feet per second of the
task minus drift, with no ride-quality cost and a small fall penalty (0.1, scaled by the time left) by default (the
episode ends at a fall, so the distance walked before it is what counts and a
gait that steps and falls beats one that stands still; `fall_cost` on the task
overrides), plus the realized stride (progress per step, capped at 0.75 x leg
length) and minus servo chatter and rate-limit costs; `step_hz_max` on a walk
task caps the cadence, which is how the walk gets longer, slower steps (the GA
will not trade speed for stride on its own, the servo slew rate sets the
stride at any cadence). Stand: fraction of time upright under periodic
shoves, minus drift (`drift_cost` per metre, default 1.0) and fan energy.
`fan_energy_cost` (0.2 by default) prices fan use so that thrust-propelled
skating on the feet scores below actual stepping. A fall is a tipped hip or
any ground contact by anything other than a foot: shins, fans, hip and core
are not built for it. Each run config lists tasks with terrains
and weights. Every individual in a generation sees the same terrain seed.

**Motor traces.** `tools/joint_trace.py <archetype or best.json> --objective walk --duration 4 --out trace.png` plots every motor's commanded and measured angle against time, one panel per motor (left solid, right dashed), and prints gait-smoothness metrics: realized stride (progress per step), servo command acceleration RMS, the share of servo command power above 2.5 x the step rate, body roll and pitch RMS.

**Throughput** (i7-12700). ~100k physics steps/s per process; the pool peaks
around 8-12 workers (~7.5x), hyperthreads add little. `tools/benchmark.py`
measures this; `generations.jsonl` logs steps/s and parallel efficiency every
generation so regressions are visible.

**Not yet.** Granular terrain (gravel is a bumpy, slippery heightfield), CMA-ES
or MAP-Elites selection, URDF export, ROS2/Gazebo validation stage.
