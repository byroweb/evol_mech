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
| Hip roll: foot under the body | `controller.act` legs block, genes group `roll` | Lateral foot target per leg = -body walk shift (`walk_shift` x half hip width, a sinusoid in phase with the step; 1.0 puts the hip over the stance foot) - `kp_roll_y` roll - `kd_roll_y` roll rate (feet toward the fall) + `kv_foot_y` x lateral body speed (the capture point: with `kp_roll_y` ~ leg drop and `kv_foot_y` ~ sqrt(drop / g) the swing foot lands where the lateral motion stops) +- `stance_w` (wider track); the hip roll angle is atan(dy / leg drop) and the sagittal IK runs in the rolled plane. |
| Tread: rubber band under a flat sole, raised front idler | `mjcf._foot`, `structure.foot_layout` | Row of 2-6 hinged rollers plus a front idler raised 0.6 r, tied by joint equalities, one velocity actuator on roller 0 with the worm's pulley torque limit. |
| Knee-mounted motor -> worm -> belt -> ankle pulley | `body/hardware.tread_drive_from_params` | Speed = motor rpm / ratio x radius (default 2.4 m/s); pulley torque = P / omega x ratio x 0.45; drive box on the front of the shin just below the knee (behind it, it hit the thigh past ~95 deg of flexion). |
| Worm does not back-drive | `controller` tread tracking | Position-tracked velocity command; zero command holds the roller angle. Parked body moves < 1 cm in 1 s. |
| No coasting: tread driven at ground speed | `controller` tread block | Command = measured ground speed along the tread + balance term + turn differential. |
| Balance | `controller.act` allocation + legs | Segway-style cascade: velocity error -> demanded accel -> tread accel (capped by tip-over and friction) -> lean, split into a foot shift (`lean_gain`) and a pitch reference; tread speed = ground speed + `kp_bal` (pitch - ref) + `kd_bal` pitch rate; fans take the remainder. |
| Suspension and levelling | `controller.act` legs block | Two-link IK on each foot target: height PD, roll levelling by unequal leg height, fore/aft foot shift on pitch error. Commands are slew-limited to the servo speed. |
| Genome | `skater/spec.py` | 63 genes in groups geometry, leg, foot, drivetrain, fan_spec, suspension, alloc, steer, gait, fan_ctrl, roll (the last legged and roll genes are interleaved at the tail). `--lock-groups` works per group. Genes were appended in that order, so an older run's genome is a prefix: pad with the defaults to re-evaluate. The `kp_roll_y` / `kd_roll_y` ranges were re-cut to 0..1 and 0..0.3 (the sign is known: feet toward the fall), so genomes from before that decode those two differently. |
| Tasks skate_flat, skate_rough, slalom, stand_perturbed | `skater/rollout.py`, `configs/skater.json` | Fitness as before: mean speed x upright fraction, ride-quality cost, lateral error, scrub and cones hit on the slalom, drift on stand, falls, energy, leg-mass penalty. The slalom course has collidable 30 cm cones on the centreline at the path's lateral extremes (every half wavelength); each cone touched costs `cone_cost` (default 0.1) and `cones_hit` is reported. Only `foot_*` geoms may touch the floor. The crab objective exists in code but is not a stage-1 task. |
| Legged tasks walk, hop, broad_jump | `skater/rollout.py`, `controller.gait_offsets`, genes group `gait`, `configs/skater_legs.json` | Treads locked (position hold), feet moved by an open-loop cycle whose shape is evolved: walk = stance foot travels back under the hip at `step_hz` over `step_len` for `step_duty` of the cycle, swing foot returns lifted `step_lift`, legs half a cycle apart, body shifted toward the stance side by `walk_shift` peaking at `shift_phase`; hop / broad jump = crouch `jump_crouch`, thrust to `jump_ext` above nominal (feet `jump_lean` behind the hip on the broad jump), tuck, settle. On locked soles the body is an inverted pendulum on the passive ankle hinge, and gravity's stiffness there (m g h, about 7 N m/rad for 2 kg at 0.35 m) exceeds the original 6 N m/rad ankle spring cap, so a legged stance fell within a second whatever the gait (the Segway treads had been hiding this in every skating task); the ankle spring range now reaches 40 N m/rad. Legged mode also asks nothing of the treads (no lean reference) and lets the fans push forward only when airborne, since a fan push against locked soles just topples the body. Pitch is then balanced by shifting the feet under the body with the legged gains `kp_foot_leg`, `kd_foot_leg`, `ki_foot_leg` and `kv_foot_x` (per m/s of forward speed). Lateral balance in a walk is the hip roll weight shift (`walk_shift`, `kp_roll_y`) with the fan over the swing foot getting `step_fan` extra throttle at full lift (feedforward on the gait phase) on top of the roll PD. Fitness: walk = progress of the feet (mean foot x, so a forward dive earns nothing) per second of the task minus drift, with no ride-quality cost (`accel_cost`, ACCEL_COST per objective: step impacts at skate's 0.02 outweighed the progress of a 0.1 m/s walk and standing still won again) and a fall penalty of 0.1 scaled by the fraction of the task left (`fall_cost`, FALL_COST per objective): the episode ends at a fall, so the distance walked before it is what counts, and a gait that steps and falls late outscores one that stands still, which the flat -0.5 had made the attractor of every legged run; with no penalty at all the winner fell over in the first quarter second, which avoids every running cost and scores zero. `configs/skater_walk.json` also prices motor energy at 0.0005 per J (a 3 W walk at the default 0.002 cost 40 % of its own progress). Walking itself: on this body (2 kg, hip 0.35-0.5 m up, 3 cm wide treads, passive ankle) the lateral fall time constant is sqrt(h/g) ~ 0.2 s and a tread tips at 1.5 cm of CoM offset, so a 60 % duty cycle with a sinusoidal weight shift at any useful step rate asks for lateral accelerations the feet cannot supply (0.4 g at 2 Hz x 4 cm) and the body rocks +-40 deg; what walks in a hand sweep (2304 gaits) is nearly straight legs (taller = slower fall), 80-90 % stance duty (the weight moves in double support), a small shift (0.2), 10 cm steps at 2 Hz and capture-point placement (`kv_foot_x` 0.2, `kp_roll_y` 0.4, `kv_foot_y` 0.15): 0.88 m in 8 s without a fall, half the nominal step speed (servo lag and sole slip). The remaining fall mode is a forward pitch runaway once the body outruns the stance sweep. `configs/archetypes/skater_walker.json` is that seed. Evolving the whole genome from it (`configs/skater_walk.json`, 96 x 50) still ended in a creeping stander: body mutations break the gait's thin basin faster than gait mutations improve it, and a light-legged stander scores within a few hundredths of the walker. With the body locked to the seed (`--lock-groups geometry,leg,foot,drivetrain,fan_spec`, `configs/skater_walk_gait.json`) the GA walks: 11.5 cm steps at the 3 Hz cap (since raised to 5 Hz), 85 % duty, 1.8 m in 8 s and 4 m in 20 s on flat ground with the fans idle; it fell within 3 s on cobblestone, so `configs/skater_walk_gait2.json` adds a cobblestone walk (weight 0.5) and seeds from that gait: the result (`configs/archetypes/skater_walker_evolved.json`) takes 11 cm steps at 4 Hz with 88 % duty and a 40 % weight shift, 4.7 m in 8 s and 11.5 m in 20 s on flat ground, 1.8-3.5 m on cobblestone before a fall (one of three terrain seeds drops it in 2 s), fans near idle. So the legged work goes body-first, gait-second, like the skating runs did. The evolved walk is a 13 cm shuffle at 4 Hz with a chattering thigh command (`tools/joint_trace.py` plots every motor against time), so the walk fitness also rewards the realized stride (progress per step, capped at 0.6 x leg length so the splits do not pay; `stride_weight`) and charges servo chatter (`chatter_cost` x the RMS angular acceleration of the eight servo commands, fan pitch included) and rate-limit slamming (`sat_cost` x the fraction of control steps with a leg servo at its slew limit). The fan pitch servos were banging between their stops at every footfall: the fast walk has a flight gap at each step, and any 'airborne' moment sent the fans into a forward push at idle throttle, which is a 90 deg pitch command; the legged push now needs FLIGHT_PUSH_S of sustained flight and the fan attitude terms fade out below FAN_AUTH_THR mean throttle. `configs/skater_walk_gait3.json` is that run: fans off, servo chatter 2817 -> 407 rad/s2, realized stride 14 -> 16 cm (the cap is 18 cm on these 30 cm legs), 5.0 m in 8 s on flat ground and, new, 3.0 m in 8 s on cobblestone without a fall (`configs/archetypes/skater_walker_evolved3.json`). Cadence stayed at 3.8 Hz and a leg servo is at its slew limit 88 % of the time. Raising the stride cap to 0.75 x leg length (`configs/skater_walk_gait4.json`) changed nothing (16.6 cm, 3.9 Hz, saturation 95 %): the cap was not binding, the servo slew rate is, so a walk task may now cap the cadence instead (`step_hz_max`, applied to the decoded gait before the rollout). `configs/skater_walk_gait5.json` caps the walk at 2.5 Hz: 19 cm strides (0.63 of the leg), 3.8 m in 8 s on flat ground and 2.0 m on cobblestone without a fall, saturation 68 %, chatter 379, clean knee and shin waveforms, the thigh still jagged from the pitch feedback (`configs/archetypes/skater_walker_evolved5.json`, which stores step_hz at the cap). Slower by a quarter for strides a sixth longer; the GA will not trade speed for stride length on its own, so the cadence cap is the knob. A run (flight between strides) is a different gait, not a faster walk: thrust is there (the hop run reached a 22 cm apex and a 50 cm broad flight before falling on landing), the fans add 0.5 g of forward push in flight, so a `run` objective was added (genes `run_hz`, `run_len`, `run_duty`, `run_lift`, `run_crouch`, `run_ext`, `run_fan`; `configs/skater_run.json`, `configs/archetypes/skater_runner_seed.json`): a bounding cycle with the stance foot sweeping back while the hip, held `run_crouch` below the parked height, recovers `run_ext` of it by toe-off, the swing foot retracted toward the hip and carried forward to land ahead of it, duty below 0.5 leaving a flight between stances, the fans spinning to `run_fan` and pushing forward while no foot is down; fitness = feet speed + 0.5 x realized stride (cap 1.2 x leg) + 0.3 x the share of the time in real flights (hip rising >= 1 cm; a fast walk already loses foot contact a fifth of the time without the hip rising, so plain airborne time rewards flicker). What the hand sweeps found: (1) the parked stance is nearly straight-legged (knee 20 deg, 0.295 of a 0.30 m reach), so there is no extension to thrust with and a run has to live in a crouch; (2) a crouched legged stance cannot be balanced: with both feet planted and a 2-4 cm crouch commanded, the body pitches over within a second at every ankle stiffness (2-25 N m/rad) and every foot-placement gain tried, forward with the walker's gains and nose-up with positive ones, while the same body upright holds for seconds; the knee tracks its command, so it is the balance, not the drive. That is also why every hop fell on landing and why the GA's bounds all fell. The run seed that survives is a gapped 2.5 Hz shuffle with 3 cm of crouch (2.0 m in 6 s, no real flights), and 40 generations of the body-locked GA from it (`runs/skater_run`) stayed one: 2.2 m in 6 s, 0.37 m/s, slower than the 2.5 Hz walk, 33 % duty on paper but the hip never rising, a leg servo at its slew limit 89 % of the time, fans off. The crouched balance turned out to be two foot problems, not a control law. (a) The passive ankle spring's rest was 'sole level at the parked crouch' and the spring is stiff enough to stand on (25 N m/rad), so bending the knee tilted the sole with the shin: keeping it flat would have needed ~10 N m of spring deflection against the ~1 N m the body's weight can hold over a half sole, so the foot rode on one edge and the only stable pose was the straight-legged one. The fix is a sole-levelling parallelogram (`controller._level_soles`, `LEVEL_SOLES`): the ankle rest follows the measured thigh and knee so the sole stays parallel to the body at any knee angle, the spring acting about that; it is a passive linkage in hardware and a joint spring reference tracked each control step in sim. It also lifts skating (flat 0.38 -> 0.50, slalom and stand up too) and only the walks drop, because they had evolved on the edge-riding foot. (b) The feet were placed with the ankle under the hip, but the CoM sits ~2 cm ahead of the hip and the sole's support is only +-6 cm about the ankle, so the parked stance stood on its front edge and fell forward; crouched, the hip drops behind the feet and it fell backward. `stance_x` (feet fore/aft of the hip) fixes that; a live CoM feed-forward from the model was tried and oscillates (the legs are a quarter of the mass, so moving the feet moves the CoM that moves the feet). Static-stance sweeps (crouch 0 / 3 / 6 cm, shoves of +-3 N for 0.2 s): with stance_x +1 cm, no pitch gain and kv_foot_x 0.1-0.3 the stance holds at every crouch within 6 deg; pitch gain in either sign makes it worse (the body pitches by rocking on the sole, so the pitch angle is not the signal, body velocity is), and a 0.3 m/s shove is recovered 8 times in 9 with kd 0.03 and kv 0.3. With the levelled sole the run sweep produces real flights for the first time: gaits that bound in place for 6 s (4-10 flights, 3-6 cm apex) and gaits that progress and fall (`configs/archetypes/skater_runner_seed2.json`, `configs/skater_run2.json`). Both walk and run also pay for swing-foot clearance now (`clear_weight`, CLEAR_WEIGHT 2.0 per m up to CLEAR_CAP 5 cm, the median ankle rise per swing, reported as `swing_clear_m`): the round-5 walker lifts its swing foot 6 mm, a shuffle that trips on a pebble. Results on the levelled foot: the run GA (`runs/skater_run2`, 40 gens, body locked) gives the first stable bound, `configs/archetypes/skater_runner_evolved.json`: 1.6 Hz, 19 cm strides, 16 real flights in 6 s with the body airborne half the time (the flight cap), 3 cm apex, 6 cm clearance, 1.8 m in 6 s (0.3 m/s, a forward hop rather than a fast run), knee 70-80 deg in the crouched stance snapping to 35-40 deg at push-off, fans pulsing to half throttle at take-off; the walk re-evolved on the new foot (`runs/skater_walk_gait6`, `configs/archetypes/skater_walker_evolved6.json`): 3.05 m in 8 s at 2.5 Hz with 15 cm strides and 3.4 cm of clearance (was 3.8 m with 0.6 cm), 2.3 m on cobblestone with a fall at 7.8 s, still climbing at generation 39. A speed round on the bound (`configs/skater_run3.json`: `speed_weight` 2 on the feet-progress term, flight weight 0.15, 8 s, 60 gens, from the run2 best) gives `configs/archetypes/skater_runner_evolved3.json`: 3.3 m in 8 s (0.42 m/s, up from 0.30), 27 cm strides (0.9 of the leg) at 1.55 Hz with a 6 cm crouch, but only 5 real flights and a 1.3 cm apex: asked for speed, the GA traded the flight for a long low lope that is still slower than the 2.5 Hz walk (0.47 m/s). On these servos the stride is bought with the crouch (the bent knee has room to extend) rather than with flight time; hop = sum of the three highest apex rises of the hip above its take-off height - drift (a per-flight bonus was gamed by a 6 mm bounce chatter, 38 'flights' in 6 s); broad_jump = longest flight distance + 0.1 x total distance. A flight is both feet off the floor for >= 40 ms with the hip rising >= 1 cm, so sole bounce and the spawn drop do not count; `n_flights`, `max_apex_m`, `max_flight_m`, `airborne_s` are reported on every task. |
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
