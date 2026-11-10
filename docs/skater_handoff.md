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
| phase1k | round 1k: 12 x 4 cm foot cap | -0.41 (20 deg rule) | 12 x 3 foot; falls to the half shove on 16/24 flat starts |
| phase1l | round 1l: + heel-toe genes (GA set them ~off) | -0.375 | 11 x 3 foot; 13/24 shove falls |
| phase1m | round 1m: cap relaxed to 14 x 4 cm, averaged pick | -0.205 (seed -0.249) | 14 x 3 foot, 12.7 cm strides, 3.1 cm clear, core 3.3 deg, fans 1.2 %; 18/24 shoved flat starts (seed 22/24) |
| **phase1m_seed** | **current best (user's pick)**: phase1j's gains on a 14 x 3 cm foot | -0.249 | 22/24 shoved flat starts, 10.9 cm strides, 3.3 cm clear, core 2.8 deg, fans 0.6 %; preferred over phase1m for shove resistance |

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

## The downward LIDAR (VL53L8CX), simulated 2026-11-09

The user picked ST's VL53L8CX (datasheet DS14161 rev 12) and asked for the module and its field only, not its
board. `evol_mech/skater/lidar.py` models it; task key `lidar` (off by default: no site, no change to any model or
score) mounts it on the core's front face, 2 cm above the core's base (`lidar_z`), pitched 25 deg forward of straight
down (`lidar_pitch_deg`), about 40 cm up on the current walker. The controller reads it every step (`ctrl.lidar`);
nothing drives off it yet. `rollout(..., lidar_cb=f)` hands each new frame to `f(sensor, t)`.

- 8 x 8 zones over a 45 x 45 deg field, 15 Hz (the part's 60 Hz is 4 x 4 only), a frame about 33 ms old when read.
- Each zone is a 3 x 3 ray bundle. Surfaces within 60 cm of each other along the beam blend into one signal-weighted
  range (a zone half on a toe reads between toe and floor); farther apart, the strongest return wins.
- 2 cm to 4 m (`lidar_max_range_m` 1.1 for daylight on a dark floor), shorter at grazing incidence.
- Noise from the datasheet's accuracy (+-11 mm under 20 cm, +-5 % beyond, at 2.7 sigma), 2 % random invalid zones,
  and a mounting tilt of up to 1 deg per run that the robot does not know.
- The grid is in scene order (row 0 far, column 0 the robot's left); the real part's lens flips both axes, and
  `lidar.zone_ids` converts to the driver's zone order.
- The robot's own toes enter the near row as they swing (they read 6-8 cm high), as they will on the build.

Footprint on the current walker (`tools/lidar_view.py`, video and sheet in runs/lidar_view/<terrain>/): from 5 cm
to about 45 cm ahead of the module, 27 cm wide at the near edge and 39 cm at the far edge. At 0.16 m/s a patch of
ground stays in view about 2.5 s, some 35 frames, before the feet reach it.

What it can and cannot see, per frame, measured height against the true ground at the zone centres:

| terrain | true relief (sd) | reading error (rms) | correlation |
|---|---|---|---|
| flat | 0.2 cm | 0.9 cm | 0.22 |
| cobblestone (6 cm stones, 1-2 cm relief) | 0.4 cm | 1.0 cm | 0.19 |
| potholes (12-30 cm, 3 cm deep) | 1.4 cm | 2.2 cm | 0.62 |

A zone covers 4-7 cm of ground, the size of a cobble, and the noise is about 1 cm at these ranges: single frames see
potholes, steps and obstacles, not individual cobbles. Fusing the ~35 frames each patch gets into a height map
should cut the noise about six times; the zone blur stays. That map is the next step before footholds.

## Slopes are the near-term target; cobbles de-prioritised (user, 2026-11-09)

Terrain `slope` (task keys `slope_deg`, `slope_heading_deg`: 0 uphill ahead, 180 downhill, 90 rising to the left,
-90 to the right): flat start, the grade blended in from 15 cm ahead over 40 cm. Falls and swing clearance are
measured from the ground below on it. Task key `height_ref: "feet"` makes the height loop hold the hip above the
planted feet (pressure-pad contact plus leg kinematics) instead of world z, which the build cannot measure and
which is wrong on any slope; on level ground it walks as before.

Best walker (phase1m_seed), 12 s walks from 4 tilted starts, no shove, height_ref feet:

| slope | uphill | downhill | across (either way) |
|---|---|---|---|
| 3 deg | 4/4 | 4/4 | 1/4 |
| 6 deg | 2/4 | 1/4 | 0/4 |
| 10, 15 deg | 0/4, falls back | 0/4, falls forward | 0/4 |

- The passive ankle spring is the uphill/downhill limit: a sole on a grade bends it and it shoves the body
  downhill. Half the stiffness walks 6 deg uphill 4/4 and downhill 2/4 with level walking intact; a quarter
  (below m g h) fails everywhere. A fixed rest angle matched to the slope tips it on the flat run-up, so the
  trim or lean must change on the slope: the controller has to know the grade.
- Across-slope walking fails from 3 deg (mostly forward falls) without slope-aware placement (next section).
- Knowing the grade: a plane fitted to one VL53L8CX frame, levelled with the IMU, reads it within about 1-2 deg
  (median per frame); a simulated MEMS IMU's complementary filter adds ~0.7 deg of tilt error. Averaged over a second, ~1 deg.

Cobbles from the feet (1 s windows, flat vs cobble, phase1m_seed walking): how often the centre of pressure
reverses direction while a foot is planted separates them 95 % of the time; the core gyro's jitter 99 %; feet
alone in a fitted classifier 92 % on held-out windows. Few cobble windows (20: the walker falls on cobbles), and
gyro jitter may partly be pre-fall wobble.

## Slope-aware foot placement (user: no powered ankle, let the controller pick foot positions; 2026-11-09)

Task key `slope_est` (off by default; every score unchanged) turns on a grade estimate and four placement terms
(controller.py, the `slope_*` keys in `__init__`): feet back on an uphill and forward on a downhill (`slope_kx`
m/rad), feet toward the downhill side across a slope (`slope_ky`), foot targets laid on the ground's plane
(`slope_follow`), and the body pitched with the grade by where the feet are put (`slope_lean`). Why the lean: the
passive parallelogram holds the sole parallel to the body, so a level body on a grade always bends the ankle
spring by the full grade; moving the feet alone cannot undo that, a body pitched with the ground can.

Best found: `slope_kx 0.15, slope_ky 0.3, slope_follow 0.5, slope_lean 0.3`, with `height_ref feet`.
12 s walks, 8 tilted starts each, no shove (upright of 8):

| grade estimate | flat | 3 up | 3 dn | 6 up | 6 dn | 10 up | 10 dn | 3 across | 6 across L | 6 across R | 10 across |
|---|---|---|---|---|---|---|---|---|---|---|---|
| none (terms off) | 8 | 8 | 8 | 3 | 3 | 0 | 0 | 1 | 0 | 0 | 0 |
| true grade (upper bound) | 8 | 8 | 8 | 8 | 6 | 1 | 1 | 8 | 5 | 4 | 0 |
| feet: plane through the last 4 footholds, tau 0.1 s | 8 | 8 | 8 | 7 | 7 | 0 | 0 | 8 | 0 | 0 | 0 |
| fused: half lidar, half feet, both axes (lidar tau 0.3 s) | 8 | 8 | 8 | 8 | 7 | 4 | 0 | 8 | 1 | 0 | 0 |

- Shoved flat starts (the walk_flat task, 24 seeds) are unharmed: 19/24 as scored (world height), 18/24 with the
  feet estimate, 20/24 fused.
- Lag is what the estimate must avoid: the true grade read 0.2 m behind the hip loses most of the gain (6 across
  5/8 -> 1/8), read 0.15 m ahead is as good as at the hip. Six footholds lagged too much; four is better.
- Estimate accuracy with the walker upright on the grade: the foothold plane is within 0.3-0.7 deg rms along and
  0.3-0.5 across; the lidar's plane 0.8-1.7 along, 0.5-0.6 across. The soles' own pitch reads 1.3 deg rms noise.
  The foothold positions come from the true body pose and leg kinematics; on the build that is IMU tilt plus the
  servo angles, so expect more noise there.
- 6 deg across still fails: the falls come where the grade starts (x ~0.5 m), where the footholds still hold the
  flat run-up and read ~1 deg of a 5 deg grade. The lidar sees the grade ahead but taking the across grade from
  it alone hurt the downhill (0/8): something in its view (the toes, the legs) skews it on a downhill. 10 deg
  downhill fails with every estimate, the true grade included.

### Lidar ground map, confirmed by the footfalls (user: footfalls point to the past; 2026-11-09)

`slope_est: "map"` (needs `lidar: 1`): every VL53L8CX frame's valid points go into a world-frame map of the last
`map_s` (1 s) of frames, less the points on the robot's own feet; the grade is a robust plane over the map from
5 cm behind to 20 cm ahead of the hip (`map_ahead`), 15 cm either side. The sensor saw that ground from up to 45 cm
ahead, so the estimate is current at the hip instead of a step or two behind it. `slope_confirm_deg` (1) holds its
across reading within that of the across grade the latest left and right footholds give: the map sees a change
first, the two current footfalls confirm it. Map points are placed with the true body pose (on the build: IMU
attitude plus leg odometry over the map's second, not modelled yet).

Estimate error with the walker upright (rms, deg): on the grade 0.3-0.45 both ways (the 1 deg mounting error adds
~0.15); where the grade starts, 1.0-1.4 across against the footholds' 1.8-2.1, and it reads the grade early.
Cross-slope walking takes 1 deg of random grade error or a 0.5 deg bias with no loss (true grade plus noise);
the estimate's behaviour where the slope starts is what matters: reading early (true grade 0.15 m ahead) or
late (0.2 m behind) both cost 6 deg crossings, which is what the footfall check fixes.

24 tilted starts each, 12 s, no shove, the placement gains above, upright of 24:

| grade estimate | flat | 3 up | 3 dn | 6 up | 6 dn | 10 up | 10 dn | 3 across | 6 across L | 6 across R | 10 across |
|---|---|---|---|---|---|---|---|---|---|---|---|
| none | 24 | 24 | 24 | 15 | 4 | 0 | 0 | 2 | 0 | 0 | 0 |
| true grade | 24 | 24 | 24 | 24 | 19 | 6 | 5 | 24 | 13 | 5 | 2 |
| feet, 4 footholds (previous commit) | 24 | 24 | 24 | 18 | 17 | 3 | 0 | 19 | 0 | 0 | 0 |
| lidar map | 24 | 24 | 23 | 23 | 16 | 7 | 1 | 21 | 6 | 2 | 1 |
| lidar map + footfall check 1 deg | 24 | 24 | 24 | 22 | 14 | 8 | 0 | 20 | 11 | 8 | 0 |

Shoved flat starts: 20/24 with the map and check (19/24 as scored). Even the true grade crosses 6 deg only about
half the time: the rest is the placement itself (bigger sideways shifts, ky 0.5-0.9, and laying the across
targets fully on the slope both made it worse). The falls come where the grade starts: the walker stalls, the
uphill foot pins on its heel, it tips back.

### The slope's start is the hard part, not the slope (user's hunch, confirmed 2026-11-09)

Task keys `slope_x0` (where the grade starts, default 0.15 m) and `slope_blend` (how long it takes to reach the
full grade, default 0.4 m). Lidar map + footfall check, 16 tilted starts each, upright of 16:

| transition | 6 across L | 6 across R | 6 dn | 10 up | 10 dn | 10 across | 15 up |
|---|---|---|---|---|---|---|---|
| 0.4 m (default), placement off | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| 0.4 m | 8 | 6 | 8 | 5 | 0 | 0 | 0 |
| 0.8 m | 11 | 9 | 15 | 13 | 1 | 2 | 3 |
| 1.5 m | 13 | 16 | 14 | 16 | 5 | 10 | 15 |
| 1.5 m, placement off | 1 | 1 | 13 | 14 | 1 | 0 | 0 |

- A settled gait first (`slope_x0` 1-2 m, the grade starting well after the walk's own start) helps only a little
  at the sharp 0.4 m knee (6 across 9/16 -> 12/16 left, 6/16 -> 7/16 right).
- Shorter steps (`step_len_scale` 0.5-0.7: the walk's pace is the step_len gene, the task's speed key does not set
  it) do not help, they hurt. So it is not the grade change per step: it is a knee about as long as the robot's
  own stride, its feet and body on different grades at once.
- Foot landing heights read off the lidar map at each target (`slope_follow_map`, `map_r`; swing foot only)
  did not help either (6 dn 8 -> 13/16 at 0.5, other cases equal or worse).
- What a sharp-knee fall looks like (true grade, 6 across, rising left): each stance on the uphill foot runs its
  centre of pressure from toe to heel and pitches the body further back (-6, -9, -15 deg), the walker stalls in
  x through those stances, and it turns toward uphill (yaw +20-30 deg) without a way to turn back (no hip yaw,
  the fans near idle).
- `slope_oracle_noise_deg` / `slope_oracle_bias_deg`: cross-slope walking takes 1 deg of noise or 0.5 deg of
  bias on the true grade with no loss.

### GA round s1 on sharp slope starts (user's pick, 2026-11-09): only the downhill gained

The slope gains are genes now (group `slope`, appended: slope_kx, slope_ky, slope_follow, slope_follow_y,
slope_lean, slope_confirm, slope_tau_lidar, and two knee terms slope_kxc / slope_kyc: feet moved by the map's
grade 15-35 cm ahead less the grade at the hip). Task keys still override them; inert without slope_est.
`configs/skater_slope_s1.json`: seed phase1m_seed + the hand gains, tasks the shoved flat walk (slope logic on),
6 across L/R, 6 down, 10 up (all 0.4 m knee, lidar map + footfall check, 2 draws each) and the perturbed stand;
17 genes free (slope + 8 balance/placement gains), 32 x 40, stopped at gen 24 (patience 10). Picked by the mean of
5 draws: -0.729 (seed about -0.85).

Unseen starts (seeds 100-123), upright of 24:

| walker | shoved flat | flat | 3 across | 6 up | 6 down | 6 across L | 6 across R | 10 up | 10 down | 10 across |
|---|---|---|---|---|---|---|---|---|---|---|
| seed + hand gains | 19 | 24 | 19 | 22 | 11 | 6 | 8 | 11 | 0 | 1 |
| s1 winner | 18 | 24 | 20 | 22 | 20 | 4 | 7 | 8 | 0 | 0 |

The genes barely moved (slope_ky 0.30 -> 0.25, slope_confirm 1.0 -> 0.8, the knee terms 0 -> 0.015): with 2
draws per slope and mutation sigma 0.05 the round was mostly noise; the downhill gain is real, the cross-slope and
10 deg uphill are not improved. Not adopted.

### Transition fixes tried by hand (2026-11-09), none works yet

`tools/slope_video.py` films a walker through a slope's start (side and rear chase views, half speed, a readout of
pitch, roll, heading, both feet's centre of pressure heel to toe, the grade estimate against the true grade).
Every sharp-start fall filmed has the same look: the stance foot's pressure runs to its heel and stays there
while the body tips back, through two or three stances.

Lidar map, the 0.4 m knee, 16 tilted starts each (upright of 16), against the map with the gene defaults
(6 across L 7, R 6, 6 dn 12, 6 up 14, 10 up 3, 3 across 14):
- Heel-toe feedback on (cop_gain 0.01-0.04, deadband 0.4-0.7): worse everywhere; 0.02 and up drops flat walking.
- Ablations: without the sideways shift cross-slopes are 0/16; without the lean 6 dn 3/16; without the
  fore-aft shift 6 dn 2/16, 10 up 0/16; without the across height-follow 6 across 10 and 9/16 but 6 dn 8/16.
- Fore-aft shift doubled (slope_kx 0.3): 10 up 3 -> 11/16, 6 across L 10/16, but 6 up 14 -> 6/16 and 6 across R
  3/16. The shift wanted grows faster than the grade (about 1.6 cm at 6 deg, 5 cm at 10): a steady-grade matter.
- `knee_boost` (placement gains x (1 + boost) while the map sees the grade changing by knee_ref_deg ahead):
  worse at 1, much worse at 2.

## Parked ideas (the user's, for later)

- After LIDAR is simulated (the module is, 2026-11-09; the fused height map is next): cobbles via terrain-aware foot placement, then skating and slalom (the heel-toe pads
  should help the skate balance), and alternate gaits (low walk on smooth ground, high steps where the map says).
- A flying variant: 6 inch props on 28 x 10 mm motors (the fan model: 21 N each, 42 N for the pair against the
  robot's 19.4 N, 2.2 x weight, about 700 W each, 88 g each). Two thrusts plus the two fan pitch servos make a
  bicopter / tiltrotor: roll by differential thrust, pitch by common tilt, yaw by differential tilt, all three
  already in the controller. Battery life at full throttle would be minutes on the 2200 mAh pack.
  The use the user has in mind: point-to-point flights. Pick a spot; the Raspberry Pi plans takeoff, a short
  efficient trajectory and landing, seconds long, energy-hungry but brief. A leg hop gives the initial vertical
  speed (the most expensive part for the fans), the legs take the landing. Rough scale from the fan model: hover
  near 450 W (46 % thrust), so 5 s of flight is about 2 kJ. The user's pack: Tattu 2200 mAh 4S1P 14.8 V 45C, XT60
  (32.6 Wh, ~117 kJ, ~94 kJ usable at 80 %): a 5 s flight is ~2 % of it, hovering lasts ~3.5 min, full throttle on
  both big fans (1.4 kW) ~1 min. 45C is 99 A, ~1.47 kW at 14.8 V: just covers both fans at full throttle, with
  little margin for the servos and electronics. The model carries the battery as 240 g (2200 mAh).

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

### A softer ankle spring (the user's idea): the biggest slope gain so far

The parallelogram keeps the sole parallel to the body, so on a grade the planted sole bends the ankle spring by
the grade and the spring torques the body (pitched back on an uphill, rolled on a cross-slope). A softer spring
gives the foot less lever on the body. phase1m_seed has ankle_stiffness 18.4 Nm/rad; the roll spring is 0.5x that
(structure.ANKLE_ROLL_STIFF). Sharp 0.4 m starts, lidar map + footfall check, gene defaults, 32 unseen starts
(seeds 200-231), upright of 32:

| ankle_stiffness | 6 up | 6 dn | 10 up | 10 dn | 6 across L | 6 across R | 10 across L |
|---|---|---|---|---|---|---|---|
| 18.4 (seed) | 31 | 22 | 10 | 0 | 13 | 9 | 1 |
| 12 | 29 | 24 | **25** | 2 | 23 | 14 | 0 |
| 9 | 27 | 21 | 10 | 0 | 28 | 28 | 3 |
| 8 | 22 | 21 | 8 | 0 | **31** | **30** | 7 |

- Below ~7 Nm/rad (m g h) the walker collapses even on flat (5: 3/16 flat, 3: 0/16).
- Flat with shoves (walk_flat, 24 starts): 17 at 18.4, 15-16 at 8-12, inside the noise; stand_perturbed 23-24/24
  everywhere.
- Cross-slopes need both springs soft: pitch 8 with the seed's roll 9.2, or roll 4 with pitch 18.4, each did little
  on 16 starts; pitch 8 with roll 4 (the gene's coupling) went to 15-16/16.
- 12 suits uphill, 8-9 suits cross-slopes. The slope placement gains were tuned at 18.4, so a GA round with
  ankle_stiffness free (and the slope genes) is the next step.
- Video: runs/slope_video/6_across6_k8_seed101.mp4 (the start that fell at 18.4 walks through).

### GA round s2 with the spring free (user's pick): it picked ~9 Nm/rad, but its retuned gains don't carry over

`configs/skater_slope_s2.json`: start at phase1m_seed with the spring at 10, ankle_stiffness and ankle_rest free
with the 17 s1 genes; sharp starts 6 across L/R, 6 up, 6 down, 10 up at 4 repeats, shoved flat walk and perturbed
stand as guards. 42 generations, training fitness -0.95 -> -0.66. The spring settled at 9.19, ankle_rest 0.027; it
also halved slope_ky (0.3 -> 0.15), cut ki_foot_leg (0.079 -> 0.012), raised slope_confirm to 2 deg and slope_kxc
to 0.17. Fresh starts (seeds 300-331, upright of 32, `runs/score_s2.log`):

| walker | shoved flat | 6 up | 6 dn | 6 across L | 6 across R | 10 up | 10 across | 3 across |
|---|---|---|---|---|---|---|---|---|
| seed (18.4) | 22 | 26 | 16 | 18 | 8 | 7 | 0 | 28 |
| seed, spring 12 | 24 | 30 | 26 | 17 | 18 | 17 | 0 | 32 |
| **seed, spring 9** | 24 | 29 | 24 | **30** | **31** | 15 | 2 | 31 |
| s2 winner | **29** | 25 | **30** | 12 | 8 | 0 | 0 | 32 |
| s2 winner, slope_ky back to 0.3 | 26 | 17 | 12 | 2 | 5 | 0 | 0 | 27 |
| s2 balance gains + hand slope gains | 23 | 0 | 4 | 0 | 1 | 0 | 0 | 27 |

Flat is 32/32 for all; 10 down 0-2 for all. The winner's genes only work together and only on its training
starts, the same story as s1: 2-4 repeats per slope task is too few to stop it fitting particular starts.
Adopted: `configs/archetypes/skater_walker_phase1m_k9.json` = phase1m_seed with the spring at 9 and the hand
slope gains (the gene defaults). phase1m_seed stays the flat walker the tests use.
