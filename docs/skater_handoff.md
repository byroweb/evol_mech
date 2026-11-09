# Skater: handoff (2026-11-08)

Written so another session, on any model, can pick the work up cold. The long record is
`docs/skater_test_plan.md` (the plan, with a dated note under every item done) and
`docs/skater_review.md` (the narrative). This page is the short version: where things stand,
how the work is done, and what is next.

## What the thing is

A fan-assisted servo biped ("skater") evolved by a GA in MuJoCo. The current body is the
round-7 pantograph leg (`configs/archetypes/skater_walker_panto7.json`): 25 mm horns, 8.1 cm
rods with 9.4 cm extensions, 7 cm strut, 31 cm reach, 34 cm hip height, 10 cm hip, feet 2.6 cm
outboard, a passive 2-DOF sprung ankle (pitch spring `ankle_stiffness`, roll spring at half of
it, both stiffer than gravity about the ankle but passive: they hold, they do not push). Every
part has a mass and MuJoCo derives its inertia from the shape; servos and leg brackets are
massless and lumped into the hip box (the phase-2 mass rule should put them at the pivots).

## Where the walkers stand (all under `servo_model: real`, lead, clamp)

Scored on seeds the GA never saw with `tools/score_round.py`; the GA's own best has
overstated every round.

| archetype | what it is | unseen-seed weighted p20 | notes |
|---|---|---|---|
| panto7 | the body, clock gait | -1.16 | 5/8 perturbed flat walks, 15 cm strides |
| phase1e | round 1e: gains on the robustness score | -0.74 | 7/8, 4/4 stands, 12 cm strides, 3.0 cm clearance, fans 1.7 % |
| phase1f | round 1f: + capture point | -0.58 (its score) | 7/8, roll 5.3 deg (lowest), but 1.0 cm clearance: seed only |
| phase1h | round 1h: + contact gait (strider form) + 3 cm clearance floor | -0.91 (its score, seed -1.10) | the item-1 walker; 10.5 cm strides sit on the floor |
| shape_long | round-1e setup as 48 x 80 | -0.71 (GA best -0.25) | steadiest walk (roll 5.1 deg, 5 % servo saturation), 2.1 cm clearance, 3/4 stands: seed only |
| phase1j | round 1j: waist linkage 1:1 (task keys waist_link 1.0, roll_score core), 3 cm floor | -0.27 (its score, seed -0.47) | the best walker: 8/8 flat, 4/4 stands, core roll 2.4 deg, 3.3 cm clearance, 11.4 cm strides, fans 0.6 %; cobbles 0/4 |

Every one falls on cobbles inside 3 s. Roll is 5-7 deg RMS on all of them against the plan's
4: the only lateral tool is the hip roll servos, which roll the body as much as they move it.

## Phase 1 status (controller), by item

1. Contact gait: done in the strider form. Task keys `gait_phase contact`, `contact_start
   instant`, `contact_anchor command`. The shuffler's form (parked start, measured anchors)
   drops a strider at 1.2 s; a true single-support shift fell at 0.75 s every time (hip roll
   shift rolls the body; the roll-levelling term lifts the foot). Single support is a body
   question (phase 2).
2. Capture point: done (`foot_placement capture`); works on the strider alone, falls when
   stacked on the contact gait untuned (-1.01 as a seed).
3. Latency lead (`servo_lead auto`) and workspace clamp (`leg_reach clamp`): done.
4. Landing mode: not started.
5. Robustness score: done. `aggregate p20`, `mass_error 0.10`, `friction_error 0.20`,
   `push 0.5`, 8/4/4 starts, plus floors `stride_min 0.10` and `clear_min 0.03`.
6. CMA-ES: not started. The case for it: the GA best wanders 0.1 after generation 20 and
   overfits its eight fixed starts (round 1h: GA best -0.29, unseen flat -0.70).

Population-shape test, done: at 3840 evaluations 192 x 20 scored -0.91 on unseen seeds,
96 x 40 (round 1e) -0.74, 48 x 80 -0.71 with a GA best of -0.25. Use 48 x 80 from now on; the
gap between the GA best and the unseen score is the real limit.

Sway (2026-11-08): commanded by the hip roll servos, not the ankle (the ankle roll spring
does nothing; the joint flexes 0.5 deg). Foot lift and torso roll are the same motion on this
body. The waist (task keys `waist_stiffness 2.0`, `waist_damping 0.1`, `waist_pivot 0.14`: the
core hung from a sprung roll pivot at its top) keeps the core within 1.7 deg on shape_long,
untuned, but the user prefers a linkage: `waist_link 1.0` (core roll on the hip box = the mean hip
roll angle, pivot at the core's base) keeps the core within 3.0-3.2 deg on both walkers with more
clearance and no extra roll-servo load. Round 1i (half shift, cheap fans, exploratory) was running
at this writing.

## Recommended next steps, in order

0. Cobbles are set aside until the downward LIDAR is simulated (user's decision, 2026-11-08): under the user's foot
   cap (12 x 4 cm, GA config gene_bounds) the big-foot fix (24 x 7 cm walks every cobble start) is out, and the
   catch step and pitch levelling (task keys, off) saved nothing. The cobble task stays in the robustness score
   at its weight so rounds stay comparable, but it is not a phase-1 exit criterion now. Round 1k re-tunes the
   gains on the capped foot.
1. Correction (2026-11-08): the GA already draws fresh robustness starts every generation (ga.run_ga passes a new
   gen_seed each generation; all individuals of a generation share it). The GA best overstates the unseen score
   because best.json keeps the luckiest evaluation of any generation (a winner's curse), not because the starts are
   memorised. A round can now stop on a plateau (config `patience`, `min_delta`) and pick its winner by the mean of
   several draws (config `final_candidates`, `final_evals`); use both from round 1m on.
2. Item 6, CMA-ES over the gain genes on the fixed body with the robustness score.
3. Phase 2 body round from phase1h, carrying: the roll-lever range (`structure.roll_lever_range`),
   the 2.3 cm extension headroom, the tread box on the shin's outer face, a mass rule with
   servos and brackets at the pivots, and whatever gives a lateral ankle strategy (actuated
   or stiffer ankle roll, wider sole, or a translating hip). That is where single support,
   the 4 deg roll target and cobbles get solved.

## Parked ideas (the user's, for later)

- After LIDAR is simulated: cobbles via terrain-aware foot placement, then skating and slalom (the heel-toe pads
  should help the skate balance), and alternate gaits (low walk on smooth ground, high steps where the map says).
- A flying variant: 6 inch props on 28 x 10 mm motors (the fan model: 21 N each, 42 N for the pair against the
  robot's 19.4 N, 2.2 x weight, about 700 W each, 88 g each). Two thrusts plus the two fan pitch servos make a
  bicopter / tiltrotor: roll by differential thrust, pitch by common tilt, yaw by differential tilt, all three
  already in the controller. Battery life at full throttle would be minutes on the 2200 mAh pack.
  The use the user has in mind: point-to-point flights. Pick a spot; the Raspberry Pi plans takeoff, a short
  efficient trajectory and landing, seconds long, energy-hungry but brief. A leg hop gives the initial vertical
  speed (the most expensive part for the fans), the legs take the landing. Rough scale from the fan model: hover
  near 450 W (46 % thrust), so 5 s of flight is about 2 kJ, a few percent of a 3S 2200 mAh pack (voltage assumed).

## How the work is done (the rules that bit)

- The user's standing priorities: grace, stability and few fans over speed; long strides over
  cadence; feet picked up (3 cm); left foot first; report the fall direction; check what a
  winner actually does; score on unseen tilted starts, never on the GA number; every round
  under `servo_model: real`; no GPU physics. The user builds the robot in GF PETG and works
  over Remote Control: every closing message stands alone and ends with one question with
  choices.
- New score behaviour goes behind a task key whose default keeps every old score.
- Multiprocessing on Windows: always run Python from a file, never a stdin heredoc. Set
  `PYTHONIOENCODING=utf-8`. Never `mujoco.mj_setConst` after scaling a body mass.
- A GA round: `.venv\Scripts\python -m evol_mech.evolve.run configs\X.json --lock-groups
  foot,fan_spec`, launched with nohup in the background with a waiter on
  `runs/<run>/generations.jsonl` reaching the generation count. 96 x 40 takes 20-60 min
  depending on what else the machine is doing. Configs clone `skater_walk_phase1e.json`.
- Watch a run: `.venv\Scripts\python tools\live_ga.py runs` (http://localhost:8765; fitness
  curve, scoreboard, ancestry, the champion of each generation filmed into `runs/<run>/live/`,
  about 250 KB a clip).
- Score a round: `tools\score_round.py <round> <seed archetype>`; it also saves the archetype.
- Tests: `.venv\Scripts\python -m pytest tests\ -q` (55 pass). Commits go to main with
  `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>` (or the model doing the work);
  `.claude/` and the two "leg layout sketch" PNGs stay untracked; `runs/` is gitignored.
- Memory file for the project: `project-skater-legged-gaits.md` in the Claude memory
  directory, kept current with every round.
