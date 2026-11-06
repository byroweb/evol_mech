# Skater skeleton (archetype B): what was built, and where it bends

Status: stage-1 skeleton, archetype B (LEO-style legs wearing skates), built
for review before any evolution run. The biped is untouched; the skater lives
in `evol_mech/skater/` and is selected by `"body": "skater"` in a run config
(`configs/skater.json`).

Render the default body, the servo layout and an 8 s skate clip with

    .venv\Scripts\python tools\skater_snapshot.py            # -> runs/skater_skeleton/
    .venv\Scripts\python tools\skater_layout.py              # -> runs/skater_skeleton/servo_layout.png
    .venv\Scripts\python tools\view.py --body skater          # interactive, --objective slalom|stand

## The archetype in one paragraph

Per leg: three metal-gear hobby servos in the hip. The roll servo (LEO's
abduction axis, fore-aft through the hip point) swings the whole leg plane
sideways (+-30 deg) so a foot can be placed under the body. The thigh servo swings the thigh.
The shin servo drives the knee through a parallelogram, so it sets the shin's
angle relative to the body, not relative to the thigh. A squat therefore moves
both the thigh and the shin servo; moving one alone swings the foot along a
four-bar arc. Below the knee there is no servo: the ankle is a passive hinge
with a torsion spring, and the only motor below the hip is the tread drive on
the shin. Two fans sit on the core, each on its own pitch servo, no roll
vectoring. Compared with Caltech LEO (abduction servo, two flexion servos, a
quadrotor on top) this keeps the same three leg axes and puts skates on the
feet. There is no hip yaw: the first build steered by yawing each leg (+-45
deg) and that servo was traded for the roll axis once walking, hopping and
landing were asked of the legs; turning is now a tread speed differential
plus opposite fan pitch, so the slalom is a lesser objective.

## Design doc -> code

| Design doc / decision | Where | Notes |
|---|---|---|
| Mass stays at the hip; every leg servo in the hip | `skater/structure.py` | Six servos (`servo_kgcm` gene -> torque, mass, speed) are added to the hip body. Thigh carries tube + parallelogram coupler mass per metre, shin carries tube + tread drive. |
| Each leg at most ~10-12 % of total mass, penalised | `structure.leg_mass_fraction`, `rollout.LEG_MASS_PENALTY` | Default body is 12.9 % per leg (tread drive 78 g, foot 70 g, shin 37 g, thigh 44 g), so it pays 0.09 fitness. The GA can shrink the tread motor and sole; see open questions. |
| Metal-gear servos, not steppers or QDD | `body/hardware.servo_from_params` | Stall 10-60 kg cm in 5 kg cm steps; mass 25 g + 1.1 g per kg cm; 60 deg in 0.15 s (7 rad/s, slew-limited in the controller); position gain = stall torque over 4 deg. |
| Hip roll, thigh, shin (via parallelogram); passive two-axis ankle; no hip yaw | `mjcf._leg` | Hinges: roll +-30 (axis along x through the hip point, + = foot to the left on both legs), thigh -120..+60, knee 10..150 deg (no link of the shin drive comes within 10 deg of straight), ankle pitch rest +-45 deg and a passive roll of +-30 deg at half the pitch spring so a rolled leg still lands the tread flat. |
| Knee via the shin-drive four-bar | `structure.shin_from_horn / horn_from_shin / drive_stops`, `controller._drive_commands`, `mjcf._leg_actuators` | The shin servo horn, coupler and a pin tab on the shin form a four-bar (`structure.SHIN_DRIVE`: shin servo 45 mm aft of the hip pitch axis, 40 mm horn, 41 mm tab at 15 deg, coupler length auto so horn and shin are parallel at the nominal crouch; the defaults of `tools/mechanism_checker.html`). The controller converts the IK shin angle to a horn angle for the commanded thigh, clamps it to the drive's hard stops (knee 10..150, no link within 10 deg of straight; a table per thigh angle built once per body), slew-limits thigh and horn, and sets the knee position actuator to the four-bar's shin for the commanded pair. Coaxial servo + equal horn / crank reduces it to LEO's parallelogram. The coupler is still drawn, not simulated as a closed loop; interference (foot vs shin, thigh vs servo) is checked in the browser tool only. |
| Ankle passive, sprung, toe-up when unloaded | `mjcf._leg` ankle joint | `stiffness = ankle_stiffness` (gene, 0.2-6 N m/rad), `springref` = sole level at the parked crouch plus `ankle_rest` toe-up, damping 10 % of stiffness. Under load the sole flattens; a step lifts the toe first. |
| Fans pitch only, independent channels | `mjcf._fan`, `controller` fans block | One pitch hinge per fan. Common pitch = allocation tilt + pitch PD; opposite pitch = yaw torque (`fan_kp_yaw`); differential throttle = roll. Sideways push, if ever wanted, is a body roll plus common thrust. |
| Feet always parallel, turning by differential | `controller.act` treads block | Each tread runs at its ground speed along the body; the tread speed difference (`steer_kp` x heading error) and opposite fan pitch (`fan_kp_yaw`) turn the body. A tank turn scrubs the rollers, which the slalom's `scrub_cost` charges. |
| Hip roll: foot under the body | `controller.act` legs block, genes group `roll` | Lateral foot target per leg = -body walk shift (`walk_shift`, a sinusoid in phase with the step so the body sits over the stance foot) - `kp_roll_y` roll - `kd_roll_y` roll rate (feet toward the fall) +- `stance_w` (wider track); the hip roll angle is atan(dy / leg drop) and the sagittal IK runs in the rolled plane. |
| Tread: rubber band under a flat sole, raised front idler | `mjcf._foot`, `structure.foot_layout` | Row of 2-6 hinged rollers plus a front idler raised 0.6 r, tied by joint equalities, one velocity actuator on roller 0 with the worm's pulley torque limit. |
| Knee-mounted motor -> worm -> belt -> ankle pulley | `body/hardware.tread_drive_from_params` | Speed = motor rpm / ratio x radius (default 2.4 m/s); pulley torque = P / omega x ratio x 0.45; drive box on the front of the shin just below the knee (behind it, it hit the thigh past ~95 deg of flexion). |
| Worm does not back-drive | `controller` tread tracking | Position-tracked velocity command; zero command holds the roller angle. Parked body moves < 1 cm in 1 s. |
| No coasting: tread driven at ground speed | `controller` tread block | Command = measured ground speed along the tread + balance term + turn differential. |
| Balance | `controller.act` allocation + legs | Segway-style cascade: velocity error -> demanded accel -> tread accel (capped by tip-over and friction) -> lean, split into a foot shift (`lean_gain`) and a pitch reference; tread speed = ground speed + `kp_bal` (pitch - ref) + `kd_bal` pitch rate; fans take the remainder. |
| Suspension and levelling | `controller.act` legs block | Two-link IK on each foot target: height PD, roll levelling by unequal leg height, fore/aft foot shift on pitch error. Commands are slew-limited to the servo speed. |
| Genome | `skater/spec.py` | 56 genes in groups geometry, leg, foot, drivetrain, fan_spec, suspension, alloc, steer, gait, fan_ctrl, roll. `--lock-groups` works per group. Groups were appended in that order, so an older run's genome is a prefix: pad with the defaults to re-evaluate. |
| Tasks skate_flat, skate_rough, slalom, stand_perturbed | `skater/rollout.py`, `configs/skater.json` | Fitness as before: mean speed x upright fraction, ride-quality cost, lateral error, scrub and cones hit on the slalom, drift on stand, falls, energy, leg-mass penalty. The slalom course has collidable 30 cm cones on the centreline at the path's lateral extremes (every half wavelength); each cone touched costs `cone_cost` (default 0.1) and `cones_hit` is reported. Only `foot_*` geoms may touch the floor. The crab objective exists in code but is not a stage-1 task. |
| Legged tasks walk, hop, broad_jump | `skater/rollout.py`, `controller.gait_offsets`, genes group `gait`, `configs/skater_legs.json` | Treads locked (position hold), feet moved by an open-loop cycle whose shape is evolved: walk = stance foot travels back under the hip at `step_hz` over `step_len`, swing foot returns lifted `step_lift`, legs half a cycle apart; hop / broad jump = crouch `jump_crouch`, thrust to `jump_ext` above nominal (feet `jump_lean` behind the hip on the broad jump), tuck, settle. Balance is then fans plus the foot-shift terms; lateral balance in a walk is the hip roll weight shift (`walk_shift`, `kp_roll_y`) with the fan over the swing foot getting `step_fan` extra throttle at full lift (feedforward on the gait phase) on top of the roll PD. Fitness: walk as skate; hop = sum of the three highest apex rises of the hip above its take-off height - drift (a per-flight bonus was gamed by a 6 mm bounce chatter, 38 'flights' in 6 s); broad_jump = longest flight distance + 0.1 x total distance. A flight is both feet off the floor for >= 40 ms with the hip rising >= 1 cm, so sole bounce and the spawn drop do not count; `n_flights`, `max_apex_m`, `max_flight_m`, `airborne_s` are reported on every task. |
| MuJoCo contact settings | `mjcf.build_mjcf` | `cone="elliptic" impratio="10" noslip_iterations="3"`; with defaults the treads crept at low force. |

## Where the implementation departs from the doc, deliberately

1. **Hip roll instead of hip yaw.** The design doc had both; the build has
   three hip servos, and the third is LEO's abduction axis, not yaw. In
   single support the roll servo carries the body weight at half the hip
   width (about 2-3 N m static for a 2 kg body on a 0.2 m track, 40-60 kg cm
   with a dynamic margin), which is the top of the servo catalogue, so expect
   the GA to narrow the hip or shorten single support. The sole follows a
   rolled leg through a passive ankle roll spring. Without yaw the treads
   always point along the body and the slalom is run as a tank turn plus fan
   yaw, so it is weighted below the other tasks in the run configs.
2. **Ankle is passive.** The doc had an ankle servo driven by pushrod. A sprung
   hinge replaces it; the sole follows the ground and the spring sets the
   toe-up rest. Consequence: there is no ankle balance strategy, so pitch is
   held by foot shift, tread force and fan pitch only.
3. **Ankle rest defaults to level, not toe-up.** With a 6 deg toe-up rest and
   the spring at 1.5 N m/rad the planted sole pre-loads the body nose-down and
   the hold drifted 3.8 m in 6 s; at 0 deg it drifts under 1 m. The gene is
   still 0-20 deg, so evolution can trade preload against step clearance.
4. **Tread force is capped by a tip-over limit, not only by friction.** Pushing
   at ground level with the centre of mass 40 cm up tips a 14 cm sole at about
   1.2 m/s^2. Beyond `TIP_MARGIN` x that the remainder goes to the fans.
5. **The shin drive is a joint actuator, not a closed loop.** The knee
   position actuator with the four-bar's target reproduces the kinematics,
   the hard stops and the servo's torque limit at the horn; the varying
   mechanical advantage of the linkage, coupler flex and pin friction are not
   modelled. Driving the knee from the measured thigh instead of the commanded
   one is closer to a rigid link but makes the two position loops chatter
   (RMS hip acceleration tripled), so the commanded thigh is used.
6. **No lateral tread compliance.** Rigid rollers, so a yawed tread scrubs
   with full friction, which the slalom's scrub cost sees.

## Numbers for the default body

| | |
|---|---|
| Total mass | 1.78 kg |
| Hip incl. six 25 kg cm servos | 0.54 kg |
| Per leg (thigh, shin + tread drive, foot) | 0.23 kg, 12.9 % |
| Leg servo | 2.45 N m, 53 g, 7 rad/s |
| Fan, each, incl. pitch servo | 37 g, 5.2 N max |
| Tread | 2.4 m/s max, 11 N max, 0.20 N m at the pulley, 78 g drive |
| Hip height in the parked crouch | 0.33 m (thigh 150, shin 150, knee 40 deg; thigh -20, shin +20 deg) |

Default-gain behaviour (seed 0, 6 s unless noted, ankle 4 N m/rad, vel_kp 3,
steer_kp 1, path_kp 0.8):

| Task | Upright | Distance | RMS hip accel |
|---|---|---|---|
| hold (0 m/s) | 6.0 s | 1.8 m drift | 5.9 |
| skate 0.6 m/s | 6.0 s | 2.1 m | 2.2 |
| skate 1.2 m/s | 6.0 s | 5.1 m | 2.1 |
| slalom 0.6 m/s, 8 s | 8.0 s | 3.2 m, 7 cm lateral drift | 5.0 |
| cobblestone 0.6 m/s | 6.0 s | 1.9 m | 4.2 |
| stand with shoves | 6.0 s | 1.0 m drift | 5.8 |

Nothing falls, but the hold and stand still creep forward: the cascade has no
integral term, so a small steady pitch offset turns into a steady tread speed.
Softer ankles (1.5 N m/rad) hold better (0.5-0.9 m) and skate worse (slalom
falls at 3 s). This is the first thing evolution should find a better point
for; a velocity integral in the controller is the alternative.

## Open questions for review

- **Hip roll.** Decided: the third hip servo is roll, yaw is gone. A fourth
  servo (yaw back) is the fallback if skating agility turns out to matter.
- **Leg mass.** 12.9 % against a 12 % target. The tread drive (78 g) is the
  bulk; a 40 W motor or a shorter sole brings it under. Or relax the target.
- **Fan vectoring.** Decided: pitch only, independent channels.
- **Crab.** With no hip yaw a crab is a fan push with the body leaned on the
  hip roll servos; the objective exists in code but is not a task.
- **Tread drive numbers.** Worm efficiency 0.45 and a 19 000 rpm motor cap are
  placeholders in `body/hardware.py`; replace with the chosen parts.
- **Servo numbers.** 25 g + 1.1 g per kg cm and 0.15 s per 60 deg are typical
  for 20-35 kg cm metal-gear servos; a chosen part number pins them.
- **Ride-quality cost** is RMS hip acceleration including the spawn drop; fine
  for ranking, crude for absolute numbers.
