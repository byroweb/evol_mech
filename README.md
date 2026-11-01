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
height below it, so the foot path is set by link lengths and that triangle. A ±30° yaw pivot carries
the core, which holds a light shell plus a dense payload block (battery and
electronics) whose position is in the genome. Two pitch-vectored fans sit on
the core. Linkages that are not rotatable crank-rockers are rejected before
simulation (`body/linkage.py`). Both fixed pivots A and B are mounted on the
hip, never the core, so the waist angle does not change the foot locus;
`pivot_b_dz` can place B above the hip box, which means a bracket rising past
the waist on a real build.

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

**Fitness** (`sim/rollout.py`). Walk: forward distance, minus lateral drift,
fall penalty and fan energy. Stand: fraction of time upright under periodic
shoves, minus drift and fan energy. Each run config lists tasks with terrains
and weights. Every individual in a generation sees the same terrain seed.

**Throughput** (i7-12700). ~100k physics steps/s per process; the pool peaks
around 8-12 workers (~7.5x), hyperthreads add little. `tools/benchmark.py`
measures this; `generations.jsonl` logs steps/s and parallel efficiency every
generation so regressions are visible.

**Not yet.** Granular terrain (gravel is a bumpy, slippery heightfield), CMA-ES
or MAP-Elites selection, URDF export, ROS2/Gazebo validation stage.
