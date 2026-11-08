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
| Legged tasks walk, hop, broad_jump | `skater/rollout.py`, `controller.gait_offsets`, genes group `gait`, `configs/skater_legs.json` | Treads locked (position hold), feet moved by an open-loop cycle whose shape is evolved: walk = stance foot travels back under the hip at `step_hz` over `step_len` for `step_duty` of the cycle, swing foot returns lifted `step_lift`, legs half a cycle apart, body shifted toward the stance side by `walk_shift` peaking at `shift_phase`; hop / broad jump = crouch `jump_crouch`, thrust to `jump_ext` above nominal (feet `jump_lean` behind the hip on the broad jump), tuck, settle. On locked soles the body is an inverted pendulum on the passive ankle hinge, and gravity's stiffness there (m g h, about 7 N m/rad for 2 kg at 0.35 m) exceeds the original 6 N m/rad ankle spring cap, so a legged stance fell within a second whatever the gait (the Segway treads had been hiding this in every skating task); the ankle spring range now reaches 40 N m/rad. Legged mode also asks nothing of the treads (no lean reference) and lets the fans push forward only when airborne, since a fan push against locked soles just topples the body. Pitch is then balanced by shifting the feet under the body with the legged gains `kp_foot_leg`, `kd_foot_leg`, `ki_foot_leg` and `kv_foot_x` (per m/s of forward speed). Lateral balance in a walk is the hip roll weight shift (`walk_shift`, `kp_roll_y`) with the fan over the swing foot getting `step_fan` extra throttle at full lift (feedforward on the gait phase) on top of the roll PD. Fitness: walk = progress of the feet (mean foot x, so a forward dive earns nothing) per second of the task minus drift, with no ride-quality cost (`accel_cost`, ACCEL_COST per objective: step impacts at skate's 0.02 outweighed the progress of a 0.1 m/s walk and standing still won again) and a fall penalty of 0.1 scaled by the fraction of the task left (`fall_cost`, FALL_COST per objective): the episode ends at a fall, so the distance walked before it is what counts, and a gait that steps and falls late outscores one that stands still, which the flat -0.5 had made the attractor of every legged run; with no penalty at all the winner fell over in the first quarter second, which avoids every running cost and scores zero. `configs/skater_walk.json` also prices motor energy at 0.0005 per J (a 3 W walk at the default 0.002 cost 40 % of its own progress). Walking itself: on this body (2 kg, hip 0.35-0.5 m up, 3 cm wide treads, passive ankle) the lateral fall time constant is sqrt(h/g) ~ 0.2 s and a tread tips at 1.5 cm of CoM offset, so a 60 % duty cycle with a sinusoidal weight shift at any useful step rate asks for lateral accelerations the feet cannot supply (0.4 g at 2 Hz x 4 cm) and the body rocks +-40 deg; what walks in a hand sweep (2304 gaits) is nearly straight legs (taller = slower fall), 80-90 % stance duty (the weight moves in double support), a small shift (0.2), 10 cm steps at 2 Hz and capture-point placement (`kv_foot_x` 0.2, `kp_roll_y` 0.4, `kv_foot_y` 0.15): 0.88 m in 8 s without a fall, half the nominal step speed (servo lag and sole slip). The remaining fall mode is a forward pitch runaway once the body outruns the stance sweep. `configs/archetypes/skater_walker.json` is that seed. Evolving the whole genome from it (`configs/skater_walk.json`, 96 x 50) still ended in a creeping stander: body mutations break the gait's thin basin faster than gait mutations improve it, and a light-legged stander scores within a few hundredths of the walker. With the body locked to the seed (`--lock-groups geometry,leg,foot,drivetrain,fan_spec`, `configs/skater_walk_gait.json`) the GA walks: 11.5 cm steps at the 3 Hz cap (since raised to 5 Hz), 85 % duty, 1.8 m in 8 s and 4 m in 20 s on flat ground with the fans idle; it fell within 3 s on cobblestone, so `configs/skater_walk_gait2.json` adds a cobblestone walk (weight 0.5) and seeds from that gait: the result (`configs/archetypes/skater_walker_evolved.json`) takes 11 cm steps at 4 Hz with 88 % duty and a 40 % weight shift, 4.7 m in 8 s and 11.5 m in 20 s on flat ground, 1.8-3.5 m on cobblestone before a fall (one of three terrain seeds drops it in 2 s), fans near idle. So the legged work goes body-first, gait-second, like the skating runs did. The evolved walk is a 13 cm shuffle at 4 Hz with a chattering thigh command (`tools/joint_trace.py` plots every motor against time), so the walk fitness also rewards the realized stride (progress per step, capped at 0.6 x leg length so the splits do not pay; `stride_weight`) and charges servo chatter (`chatter_cost` x the RMS angular acceleration of the eight servo commands, fan pitch included) and rate-limit slamming (`sat_cost` x the fraction of control steps with a leg servo at its slew limit). The fan pitch servos were banging between their stops at every footfall: the fast walk has a flight gap at each step, and any 'airborne' moment sent the fans into a forward push at idle throttle, which is a 90 deg pitch command; the legged push now needs FLIGHT_PUSH_S of sustained flight and the fan attitude terms fade out below FAN_AUTH_THR mean throttle. `configs/skater_walk_gait3.json` is that run: fans off, servo chatter 2817 -> 407 rad/s2, realized stride 14 -> 16 cm (the cap is 18 cm on these 30 cm legs), 5.0 m in 8 s on flat ground and, new, 3.0 m in 8 s on cobblestone without a fall (`configs/archetypes/skater_walker_evolved3.json`). Cadence stayed at 3.8 Hz and a leg servo is at its slew limit 88 % of the time. Raising the stride cap to 0.75 x leg length (`configs/skater_walk_gait4.json`) changed nothing (16.6 cm, 3.9 Hz, saturation 95 %): the cap was not binding, the servo slew rate is, so a walk task may now cap the cadence instead (`step_hz_max`, applied to the decoded gait before the rollout). `configs/skater_walk_gait5.json` caps the walk at 2.5 Hz: 19 cm strides (0.63 of the leg), 3.8 m in 8 s on flat ground and 2.0 m on cobblestone without a fall, saturation 68 %, chatter 379, clean knee and shin waveforms, the thigh still jagged from the pitch feedback (`configs/archetypes/skater_walker_evolved5.json`, which stores step_hz at the cap). Slower by a quarter for strides a sixth longer; the GA will not trade speed for stride length on its own, so the cadence cap is the knob. A run (flight between strides) is a different gait, not a faster walk: thrust is there (the hop run reached a 22 cm apex and a 50 cm broad flight before falling on landing), the fans add 0.5 g of forward push in flight, so a `run` objective was added (genes `run_hz`, `run_len`, `run_duty`, `run_lift`, `run_crouch`, `run_ext`, `run_fan`; `configs/skater_run.json`, `configs/archetypes/skater_runner_seed.json`): a bounding cycle with the stance foot sweeping back while the hip, held `run_crouch` below the parked height, recovers `run_ext` of it by toe-off, the swing foot retracted toward the hip and carried forward to land ahead of it, duty below 0.5 leaving a flight between stances, the fans spinning to `run_fan` and pushing forward while no foot is down; fitness = feet speed + 0.5 x realized stride (cap 1.2 x leg) + 0.3 x the share of the time in real flights (hip rising >= 1 cm; a fast walk already loses foot contact a fifth of the time without the hip rising, so plain airborne time rewards flicker). What the hand sweeps found: (1) the parked stance is nearly straight-legged (knee 20 deg, 0.295 of a 0.30 m reach), so there is no extension to thrust with and a run has to live in a crouch; (2) a crouched legged stance cannot be balanced: with both feet planted and a 2-4 cm crouch commanded, the body pitches over within a second at every ankle stiffness (2-25 N m/rad) and every foot-placement gain tried, forward with the walker's gains and nose-up with positive ones, while the same body upright holds for seconds; the knee tracks its command, so it is the balance, not the drive. That is also why every hop fell on landing and why the GA's bounds all fell. The run seed that survives is a gapped 2.5 Hz shuffle with 3 cm of crouch (2.0 m in 6 s, no real flights), and 40 generations of the body-locked GA from it (`runs/skater_run`) stayed one: 2.2 m in 6 s, 0.37 m/s, slower than the 2.5 Hz walk, 33 % duty on paper but the hip never rising, a leg servo at its slew limit 89 % of the time, fans off. The crouched balance turned out to be two foot problems, not a control law. (a) The passive ankle spring's rest was 'sole level at the parked crouch' and the spring is stiff enough to stand on (25 N m/rad), so bending the knee tilted the sole with the shin: keeping it flat would have needed ~10 N m of spring deflection against the ~1 N m the body's weight can hold over a half sole, so the foot rode on one edge and the only stable pose was the straight-legged one. The fix is a sole-levelling parallelogram (`controller._level_soles`, `LEVEL_SOLES`): the ankle rest follows the measured thigh and knee so the sole stays parallel to the body at any knee angle, the spring acting about that; it is a passive linkage in hardware and a joint spring reference tracked each control step in sim. It also lifts skating (flat 0.38 -> 0.50, slalom and stand up too) and only the walks drop, because they had evolved on the edge-riding foot. (b) The feet were placed with the ankle under the hip, but the CoM sits ~2 cm ahead of the hip and the sole's support is only +-6 cm about the ankle, so the parked stance stood on its front edge and fell forward; crouched, the hip drops behind the feet and it fell backward. `stance_x` (feet fore/aft of the hip) fixes that; a live CoM feed-forward from the model was tried and oscillates (the legs are a quarter of the mass, so moving the feet moves the CoM that moves the feet). Static-stance sweeps (crouch 0 / 3 / 6 cm, shoves of +-3 N for 0.2 s): with stance_x +1 cm, no pitch gain and kv_foot_x 0.1-0.3 the stance holds at every crouch within 6 deg; pitch gain in either sign makes it worse (the body pitches by rocking on the sole, so the pitch angle is not the signal, body velocity is), and a 0.3 m/s shove is recovered 8 times in 9 with kd 0.03 and kv 0.3. With the levelled sole the run sweep produces real flights for the first time: gaits that bound in place for 6 s (4-10 flights, 3-6 cm apex) and gaits that progress and fall (`configs/archetypes/skater_runner_seed2.json`, `configs/skater_run2.json`). Both walk and run also pay for swing-foot clearance now (`clear_weight`, CLEAR_WEIGHT 2.0 per m up to CLEAR_CAP 5 cm, the median ankle rise per swing, reported as `swing_clear_m`): the round-5 walker lifts its swing foot 6 mm, a shuffle that trips on a pebble. Results on the levelled foot: the run GA (`runs/skater_run2`, 40 gens, body locked) gives the first stable bound, `configs/archetypes/skater_runner_evolved.json`: 1.6 Hz, 19 cm strides, 16 real flights in 6 s with the body airborne half the time (the flight cap), 3 cm apex, 6 cm clearance, 1.8 m in 6 s (0.3 m/s, a forward hop rather than a fast run), knee 70-80 deg in the crouched stance snapping to 35-40 deg at push-off, fans pulsing to half throttle at take-off; the walk re-evolved on the new foot (`runs/skater_walk_gait6`, `configs/archetypes/skater_walker_evolved6.json`): 3.05 m in 8 s at 2.5 Hz with 15 cm strides and 3.4 cm of clearance (was 3.8 m with 0.6 cm), 2.3 m on cobblestone with a fall at 7.8 s, still climbing at generation 39. A speed round on the bound (`configs/skater_run3.json`: `speed_weight` 2 on the feet-progress term, flight weight 0.15, 8 s, 60 gens, from the run2 best) gives `configs/archetypes/skater_runner_evolved3.json`: 3.3 m in 8 s (0.42 m/s, up from 0.30), 27 cm strides (0.9 of the leg) at 1.55 Hz with a 6 cm crouch, but only 5 real flights and a 1.3 cm apex: asked for speed, the GA traded the flight for a long low lope that is still slower than the 2.5 Hz walk (0.47 m/s). On these servos the stride is bought with the crouch (the bent knee has room to extend) rather than with flight time. Checked over 30 s on flat ground (four seeds), the round-6 walk holds (9.2 m, no fall) while the round-2 bound falls at 10 s and the round-3 lope at 17 s, and every gait falls within 2-6 s on cobblestone: a 6-8 s task never tests whether a run's island of stability holds. So the legged work moved to a speed course (objective `course`, `configs/skater_course.json`): a task-level gait schedule (`profile`, default stand 2 s, walk 10 s, run 10 s, walk 10 s, stand 4 s) with every segment and every gait change scored on its own (`rollout._course_scores`, keys `seg<k>_<gait>.*` and `trans<k>_<gait>.*`): a walk or run segment on its steady window (the segment less the first 2 s after a change) with its own speed, stride, flight and clearance weights, weighted by its share of the stepping time; a stand on how far the feet move and the pitch RMS after the change (the stop); each change on the peak pitch over the 2 s after it (`trans_weight` 0.3 x (1 - peak / 0.5 rad), nothing if the body is down by the end of the window) with the time to 90 % of the new segment's speed reported (`t90_s`, the start). Energy and drift costs are scaled to an 8 s equivalent (COST_REF_S) so the long task is not priced out of using its fans. In the controller the walk and the run are one stepping cycle (`SkaterController._leg`: stance sweep over the duty, lifted return, crouch and extension; the walk is crouch 0), so a change blends the cycle's parameters (cadence, sweep, duty, lift, crouch, extension) with a walk/run mix and scales the amplitudes with a stepping amplitude (0 standing), both slewed over the `trans_s` gene, through one shared phase that keeps the legs half a cycle apart and parks at a left touchdown while stopped. Blending the two foot paths instead (tried first) had both feet lifted with no thrust at half-blend and toppled the body at every change. The run has its own fore/aft balance genes (`run_stance_x`, `run_kv_x`, `run_kp_leg`), mixed in with the blend. What the course shows: the round-3 lope falls at every change whichever walk precedes it and even when started from a stand; the round-6 walker's balance genes (`kd_foot_leg`, `ki_foot_leg`) carried into the run topple the bound at the walk -> run change; the round-2 bound completes the whole course at a 1 s blend (changes at 6 / 16 / 15 / 7 deg of pitch, 38 % of the run in flight, 36 s upright, a 1 cm stop). The seed is that bounder with the round-6 walker's step cycle (`configs/archetypes/skater_courser_seed.json`: walks 0.38 m/s, runs 0.18 m/s with 43 % flight, course fitness 0.79). The first course round (`runs/skater_course`, 40 gens, 11 s each) gives course fitness 1.27 (walks 0.41 m/s with 18 cm strides, run 0.22 m/s with 36 % flight and a 7.6 cm crouch, changes at 6 / 8 deg, a 30 s run segment held with 53 % flight), but it reproduces only from the raw genome: the archetype's phys -> genome round trip moves the genome by 1e-16 and that copy falls at the run -> walk change at 16 s. Under a random initial tilt (`pose_noise_deg`, roll and pitch uniform within the bound) it completes the course 6-7 times in 8 at +-1-4 deg, the seed 3-4 in 8, the plain bounder 4-6 in 8, with the falls clustered at the run -> walk change (14-22 s) and the stop (34.5 s): every gait so far lives on a knife edge that one deterministic rollout cannot see. So a task may now set `repeats` (rolled out that many times on different seeds, fitness the mean, metrics from the worst-upright one) and `pose_noise_deg`; `configs/skater_course2.json` evolves the course best under +-1.5 deg x 3 repeats (40 gens, 32 s each): it returns the same genome (`configs/archetypes/skater_courser_evolved2.json`, which also stores the raw genome), still 7 in 8 at +-1-2 deg and 6 in 8 at +-4 deg, with a 30 s run segment held 1-2 times in 4; three repeats cannot tell 7/8 from 8/8 and 40 generations found no wider basin. The falls are at the run -> walk change and the stop, so those two are the next targets (a change at a touchdown rather than on a timer, and a stop that keeps the walk's balance a cycle longer). The walk score then changed aim: speed is not what a walk is for, grace, stability and a quiet fan are, so the walk's gait score is now the realized stride (weight 3) plus a little speed (0.3), multiplied by a grace factor 1 / (1 + sway/0.2 rad + bob/2 cm + surge/0.15 m/s + servo saturation/2) built from the body's pitch + roll RMS, the hip height and forward-speed standard deviations and the share of the time a leg servo is at its slew limit (reported as `grace`, `sway_rad`, `bob_m`, `surge_ms`, `servo_sat_frac`), and the fans' mean throttle is charged on top of their energy (`fan_duty_cost` 1.0, reported as `fan_duty`), with the clearance bonus added outside the factor at weight 4 (the first grace round had it inside at 2 and traded the lift away: a 1.7 Hz walk with 8 mm of clearance that fell on cobblestone in 2 s); the course's walk segments are scored the same way. Multiplicative rather than subtractive because a subtractive smoothness cost made the stander win again (its gait score is nil, so grace cannot help it). Measured on the round-6 walker: sway 0.11 rad (roll 4.6 deg RMS, the weight shift), bob 3 mm, surge 0.06 m/s, a leg servo at its slew limit 89 % of the time (the thigh's pitch feedback, not the gait's speed: the 40 kg cm servo slews 7 rad/s, a 2 m/s foot sweep), grace 0.39, fan duty 6 %; with the fans off the same gait scores higher (no duty, no energy, the fan pitch servos' chatter gone: 735 -> 233 rad/s2). `configs/skater_walk_gait7.json` evolves the round-6 walker under this score, 12 s on flat ground from +-2 deg tilted starts x 2 repeats, 8 s on cobblestone, the perturbed stand. On the run: it is a hop, not a sprint. The round-2 bound's 22 flights in 8 s are 6 cm long at a 2 cm apex (hip bob 2.8 cm), so half the time is spent in the air going nowhere and the run (0.28 m/s) is slower than the 2.5 Hz walk (0.38-0.41 m/s); the flight share pays 0.15 of fitness whatever the flight covers, the leg extension is along the leg (vertical) with only `run_stance_x` to tilt it, and the fans push little (`run_fan` 0.01-0.08). So the run score changed (`configs/skater_run4.json`): speed (weight 2), stride and the mean flight length (per m, up to a leg length) are all gated by the flight share (min(1, share / 0.15): flight is the ticket, not the prize), the flight-share term itself is gone, the hip's bob is charged (2 per m), and the run has a `run_lean` gene that shifts the whole stance sweep behind the hip so the toe-off extension pushes forward (as `jump_lean` does for the broad jump); the course's run segments score the same way. Two conventions were added for consistency: every gait starts with the left foot's step (the cycle reference is still the left touchdown, but every gait starts at u = duty, the left swing start, and a course parks there while stopped), and a course stop is a braking step: the gait keeps stepping until the left foot's next touchdown, from there the cycle rate decays linearly to zero over about `trans_s` (a whole number of cycles plus the half that ends with the right foot's landing; the stance sweep is the body's speed, so the body decelerates over its planted feet) while the right foot's last swing lands beside the left instead of ahead of it, and then the amplitude fades with the feet together. The previous evolved course gaits fall under the new start and stop (knife edge: the seed falls in the run at 14 s from one tilt and at the stop from another; the round-2 best at the run -> walk change), so `configs/skater_course3.json` re-evolves the round-2 best under them. The grace factor also gained the weave (lateral hip position standard deviation, reference 2 cm): the first grace round had bought its calm servos with a cowboy walk, a 5.4 cm track (`stance_w` near its 6 cm cap) and a 0.75 weight shift, 7.6 deg of roll RMS against the round-6 walker's 4.6; `configs/skater_walk_gait8.json` re-evolves it with the weave charged. Two more lessons from those rounds: the weave must be the excursion about a moving average (1.5 s), not the plain standard deviation of y, which a 0.5 m drift over 12 s dominated; and under the grace score a completed 12 s walk is worth only ~0.15, so at a fall cost of 0.1 a gait that fell at 1 s (paying no chatter, fan or sway) won the first weave round, and the walk's fall cost is now 0.3 (scaled by the time left as before). Results. Run round 4 (`configs/archetypes/skater_runner_evolved4.json`, 50 gens, 10 s from +-1 deg x 2): 0.44 m/s (4.4 m in 10 s, up from 0.18, and faster than the 2.5 Hz walk for the first time), 28 cm strides at 1.56 Hz, 14 flights of 8.6 cm at a 1.6 cm apex, 26 % of the time in flight, holds all four tilted starts (the round-2 bound fell on one), `run_lean` only 8 mm: the speed came from a longer sweep (16 cm) and a lower bob, not from the lean, and the flights are still short, so it is a fast bound rather than a sprint. Walk round 8 (`skater_walker_evolved8.json`): 1.43 Hz, 10 cm strides, 0.15 m/s (at the speed gate), 4.9 cm of clearance, fans off entirely (step_fan and fan_base at 0), roll RMS 6.0 deg (from 7.6), weave 2.1 cm (from 3.0), chatter 193, saturation 24 %; the track is still 5.2 cm wide (`stance_w` near its cap) with a 0.67 weight shift, so the cowboy is only half cured: on a 14 cm hip with 3 cm soles the body has to carry its mass over the stance foot during single support, and the GA buys that margin with a wide track rather than a narrow roll. A sweep of the round-8 gait with the track fixed and the weight shift stepped (four tilted starts each) says the narrow track is viable (feet under the hips, shift 0.6: 12 s upright on all four, 0.09 m/s, grace 0.32) but that the roll is set by the shift, not the track (0.11-0.13 rad RMS at a 0.6-0.8 shift whatever the track), a shift of 0.4 falls on tilted starts at every track and a shift of 0.2 does not walk (2-3 cm steps). So the grace factor also charges the measured track (the feet's mean lateral separation, reported as `track_m`) beyond the hip width, TRACK_REF 5 cm (`track_ref`), on the plain walk and the course's walk segments; `configs/skater_walk_gait9.json` re-evolves the round-8 walker with the track charged. Its first run won by marching in place: 2 cm of progress in 12 s with the feet lifted 6.3 cm and the track as wide as ever, because the clearance bonus (0.2 at the cap) sat outside both the grace factor and the speed gate while a real walk's gated, graced gait score is worth a few hundredths; the clearance bonus is now scaled by the same progress gate (plain walk and course walk segments) and the round was rerun. Results: round 9 (`configs/archetypes/skater_walker_evolved9.json`) narrows the track from 22.8 to 19.7 cm (`stance_w` 0.052 -> 0.039) with 12.8 cm strides at 0.19 m/s, 5.5 cm of clearance and the fans at 1 %, but the roll RMS rises from 6.0 to 6.9 deg and a leg servo saturates 51 % of the time; round 10 (`configs/skater_walk_gait10.json`, `stance_w` locked at 0: feet under the hips, 12.8 cm track, `skater_walker_evolved10.json`) walks 8 cm strides at 0.10 m/s with 8.4 deg of roll RMS and falls on one of four tilted starts. So the track was never the roll: the roll is the weight shift. A 4 cm wide sole holds ~0.4 N m of roll moment before it tips, the ankle roll is a passive spring, so the hip cannot translate over the planted foot and the only way to put the mass over the stance foot is to roll the torso about the hip with the hip roll servo (~10 deg peak for a 5 cm shift); a narrower track makes the lever worse, not better. A level torso needs a body change: wider soles (the moment scales with the half width; a constant, SOLE_HALF_W 2 cm, not a gene yet), an ankle roll servo, a narrower hip (`hip_width` down to 8 cm halves the shift) or the fans' differential throttle, which the grace score's fan duty charge switched off. The jerk is the servo model: stiff position servos with a 7 rad/s command slew, pinned at the slew limit half the time by a linear stance sweep with corners at touchdown and toe-off; cosine-blended foot paths and a stronger saturation charge are the levers there. Three one-lever rounds from the round-9 walker under the grace score (`configs/skater_walk_sole.json`: the tread width free, a new gene `tread_w` 3-10 cm that sets the rollers' length, the sole plate's width, the roller mass and the hip-clearance check, the floor contact being the rollers and not the plate; `skater_walk_hip.json`: `hip_width` free; `skater_walk_fans.json`: fan duty uncharged, fan energy 0.05) all left the lever where the seed had it (tread 3 cm, hip 13.9 cm, fans at 1-3 % duty) with their bests found by generation 14 and the roll no better: 40 generations at sigma 0.05 do not move a body gene against the gait co-adapted to it, and a wider tread also pays the leg-mass penalty for its heavier rollers. Direct sweeps on the round-9 gait: a 10 cm tread with a 0.4 shift walks with 2.2 deg of roll RMS but at 0.05 m/s; a 10 cm hip takes the same gait from 7.0 to 5.4 deg. So each lever is re-run forced (`skater_walk_sole2.json`: tread locked at 8 cm; `skater_walk_hip2.json`: hip locked at 10 cm; `skater_walk_fans2.json`: the fans seeded with roll authority, fan_kp_roll -0.5 and fan_base 0.1, since stronger gains fought the gait's own roll and toppled it), the gait alone evolving. Forced, the levers still did not move the roll (four tilted starts each, against the round-9 walker's 6.9 deg): the 8 cm tread 7.2 deg at 0.19 m/s with 15 cm strides (`skater_walker_sole2.json`), the 10 cm hip 6.4 deg at 0.15 m/s (`skater_walker_hip2.json`; the seed gait on that hip had 5.2 deg but scored lower), the fans 8.2 deg at 11 % duty (`skater_walker_fans2.json`), each best still shifting the body 0.7-0.9 of the half hip width: the GA stays in the seed gait's basin, where the roll is the shift. `skater_walk_sole3.json` and `skater_walk_hip3.json` seed the other basin (a 0.4 shift with the feet under the hips, the sweep's 2.2 deg point) on the 10 cm tread and the 10 cm hip. Result: the hip width is the lever. On the 10 cm hip the GA keeps the small shift (0.44, feet under the hips, a 10.8 cm track) and the roll RMS falls from 6.9 to 4.1 deg with the weave at 0.9 cm, grace 0.355 (the best of any walk so far), 0.17 m/s with 10 cm strides at 1.7 Hz and 82 % duty, all four tilted starts upright (`configs/archetypes/skater_walker_hip3.json`); its swing clearance is 2.5 cm, half the round-9 walker's. On the 10 cm tread the GA left the small-shift seed, climbed back to a 0.74 shift and 7.9 deg (`skater_walker_sole3.json`): the wide foot does not hold the leg leaned over it through the passive ankle roll spring, so it buys nothing the shift cannot. The fans cost roll rather than saving it. So a level torso on servos and a passive ankle comes from a narrow hip, which halves the shift the mass must make, and the next body rounds should carry hip_width 0.10. The pantograph leg (`leg_drive` 1, `skater_design.md` "Leg drives", the user's sketch in `docs/leg_layout_sketch.png`) got its first walk on that hip: round 1 (`configs/skater_walk_panto.json`) was won by a stander, hence the walk's `stand_cost`; round 2 (`skater_walk_panto2.json`, 40 gens from the seed with the leg lengths, roll lever, horn stiffness and gait free) gives `configs/archetypes/skater_walker_panto2.json`: 2.1 m in 12 s (0.18 m/s), 11 cm strides at 1.57 Hz with 79 % duty, 5 cm of swing clearance (twice the hip3 walker's), roll RMS 4.4 deg against hip3's 4.1, weave 1 cm, grace 0.30 against 0.35, a leg servo at its slew limit 48 % of the time (the same), all four tilted starts upright, fitness -0.11 against -0.06 only because its leg-mass penalty is higher (0.28 against 0.18: rods, links and pins outweigh two tubes, and the GA already took the rods to their 8.8 cm floor). The body it chose: horns 5.6 cm (near the 6 cm cap), rods 8.8 + 3.7 cm, strut 6.1 cm, servos 3.3 cm apart, crouch 1.17 rad (the leg works well flexed, where the horn's leverage on the foot is finite), a roll lever of 1.3, and horn stiffness 10^2.83 = 680 N m/rad, in other words rigid: given the choice the GA did not take series elasticity in 40 generations. It should have: sweeping that walker's horn stiffness with nothing else changed (four tilted starts each), 1000 -> 316 -> 100 -> 50 -> 25 -> 12.6 -> 6.3 -> 3.2 N m/rad gives saturation 50 -> 45 -> 34 -> 28 -> 27 -> 24 -> 23 -> 16 %, servo chatter 444 -> 416 -> 308 -> 229 -> 209 -> 203 -> 194 rad/s2, grace 0.305 -> 0.311 -> 0.320 -> 0.334 -> 0.345 -> 0.367 -> 0.341, roll RMS 4.5 -> 4.4 -> 4.3 -> 4.2 -> 4.0 -> 3.6 -> 3.7 deg, with the speed up to 0.20 m/s down to 25 N m/rad; below that the swing clearance goes (5 -> 3 cm at 12.6, the foot drags at 6.3, and at 3.2 N m/rad the spring is softer than the 7 N m/rad the body's weight asks of a leg and it falls within 5 s). Fitness peaks at 50 (-0.06 against -0.10 rigid), the difference the GA's two noisy repeats could not see. The soft horn does what the series spring was meant to do: the servo no longer slams its slew limit at every corner of the foot path because the horn takes the step. `configs/skater_walk_panto3.json` locks the horns at 25 N m/rad (the grace-best point with the clearance still there) and re-evolves the gait and lengths around them, from the round-2 walker. Result (`configs/archetypes/skater_walker_panto3.json`, four tilted starts, against the rigid round-2 walker): a leg servo at its slew limit 12 % of the time (from 48 %), servo chatter 181 rad/s2 (from 429), 13.7 cm strides at 1.38 Hz (from 11.2: the longest strides of any grace walk), 0.19 m/s, 5.7 cm of clearance, fans at 1 %, all four starts upright, fitness -0.02 (from -0.11, the best pantograph yet); but the roll RMS rose from 4.4 to 5.6 deg because the GA took the weight shift from 0.58 to 0.69 and the grace factor stayed at 0.30, the quieter servos bought and spent on more sway. The body: 6 cm horns (the cap), rods 8 + 4 cm, a 7.5 cm strut, crouch 1.29 rad, roll lever 1.2. So the series spring removes the jerk (it was the servo slamming its slew limit at the foot path's corners, and a horn that gives 1 N m per 2.3 deg lets the servo lead the load instead), and the roll stays what it was before: the shift. The horn length pinned at its 6 cm cap says the GA wants more foot per horn degree; a longer cap is the next lever to open. So on the first pass the pantograph matches the four-bar walk in grace and roll, lifts its feet twice as high, and is heavier; the walk is in the same basin as hip3 (a 0.58 shift) rather than something new. The first course round under the new start and stop completed every tilted start (changes at 2 / 7 / 8 / 2 deg, a 0.8 cm stop) by shrinking its walk segments to a 3 cm shuffle at 0.03 m/s, hence the walk speed gate (WALK_SPEED_GATE, the gait score scaled by min(1, speed / 0.15 m/s), on the plain walk and the course's walk segments alike). With the gate, course round 3 (`configs/archetypes/skater_courser_evolved3.json`, 40 gens, +-1.5 deg x 2 repeats) completes all four tilted starts (the round-2 best did two): walks at 0.16 and 0.14 m/s with 5 cm steps at 2.4 Hz (slow and short, sitting on the gate), runs at 0.27 m/s with 23 % flight (5 cm flights), changes at 2 / 8 / 11 / 4 deg of pitch and a braking stop that leaves the feet within 1 mm; its run is a different, slower gene set from the plain run's round 4. So the left-first start and the braking stop work and are robust, and the course walk is now the least graceful thing on it: short steps at a high cadence were the GA's safe answer to the long task;  hop = sum of the three highest apex rises of the hip above its take-off height - drift (a per-flight bonus was gamed by a 6 mm bounce chatter, 38 'flights' in 6 s); broad_jump = longest flight distance + 0.1 x total distance. A flight is both feet off the floor for >= 40 ms with the hip rising >= 1 cm, so sole bounce and the spawn drop do not count; `n_flights`, `max_apex_m`, `max_flight_m`, `airborne_s` are reported on every task. |
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
- **Pantograph leg.** The second leg drive (`leg_drive` 1, see
  `skater_design.md` "Leg drives") is in and buildable. Its seed
  (`configs/archetypes/skater_walker_panto_seed.json`: the hip3 body and gait
  on 5 cm horns, 12 + 6 cm rods, a 5 cm strut, crouch 1.0, the roll and
  height derivative gains zeroed because near full extension 5 mm of leg
  length is 7 deg of horn and hip3's `kd_roll_lvl` chattered the horns at the
  slew limit) walks 0.97 m in 12 s at 5.2 deg of roll RMS. The first round
  (`configs/skater_walk_panto.json`, leg lengths, lever, horn stiffness and
  gait free) was won by a stander (`skater_walker_panto.json`: 0 stride, 0.7
  deg of roll, servos quiet, rods shrunk to the mass penalty's liking): under
  the grace score a clumsy early walk pays more in chatter, energy and the
  leg-mass penalty than its gated gait score earns, and the seed scored below
  it. So a walk task now charges standing still like a fall at the start
  (`stand_cost` 0.3 x the shortfall of the speed gate) and round 2
  (`skater_walk_panto2.json`) reruns from the seed; its result is in the
  legged row above. The roll pivots then moved to the hip centre on this body
  (the sketch's two bolts; the bracket is the lever, so the roll ratio is
  derived from the hip width and the horn, 0.64 on the 10 cm hip), which the
  soft-horn walker survives on all four tilted starts (fitness -0.02 to -0.05
  against -0.02, roll 5.3 deg). Round 4 (`configs/skater_walk_panto4.json`,
  40 gens from that walker with the hip width free, 2026-11-07) took the hip
  to 29 cm (`skater_walker_panto4.json`, derived roll ratio 2.2: the long
  bracket gears the roll servo up): it reports a GA fitness of 0.43 but on
  four tilted starts scores 0.01 against the 10 cm walker's -0.03, and the
  walk is worse where it matters: roll RMS 9.5 deg against 5.3, grace 0.22
  against 0.30, track 30 cm against 9 (it drifts sideways), fan duty 2.8 %
  against 0.9, with the same 12.5 cm stride and 7.5 cm clearance. So the
  one-lever result holds on this body too: the narrow hip is the roll fix,
  and `hip_width` 0.10 stays locked in later rounds (the wide hip's fitness
  came from the tilted-start seed, not from a better walk; check the winner).
  The horn was then capped at 25 mm (a real servo horn; the lever ratio is
  the rods' to find). Under the cap the 6 cm horn archetypes do not build;
  a random re-search of the balance genes on a short-rod, long-extension
  body gave `skater_walker_panto5_seed.json` (2.4 m in 12 s, 14.8 cm
  strides, roll 3.0 deg, grace 0.37 on the two un-tilted starts, but it
  falls on two of four tilted starts). Round 5 (`skater_walk_panto5.json`,
  hip locked at 10 cm, horns soft, leg lengths and gait free) gives
  `skater_walker_panto5.json`: the GA shortened the extensions (16 -> 9 cm)
  and the strut, walks all four tilted starts, 1.9 m in 12 s with 15.8 cm
  strides at 1.0 Hz (the longest, slowest walk yet), 3.8 cm of clearance,
  roll 5.7 deg, fitness -0.16, but a leg servo at its slew limit 62 % of
  the time and grace 0.20: with the shorter extensions each step is more
  horn sweep, and the 7 rad/s servo is the limit. Against the 6 cm horn's
  round 3 (fitness -0.02, grace 0.30, saturation 12 %) the 25 mm horn is
  behind on everything but stride length; the cap turns the servo's speed,
  not its torque, into the constraint. Round 6
  (`configs/skater_walk_panto6.json`, `leg_mass_penalty` 0: the distal-mass
  penalty switched off for the round, from walker_panto5) tested whether
  that penalty was what kept the extensions short. It was not: the GA
  number rose by exactly the penalty it no longer paid (0.57 against
  -0.16) and the body stayed where it was (extensions 9.5 cm, strut 6 cm,
  leg 15.5 % of the mass against 15.4 %), with the same 16 cm strides at
  1 Hz, 4.6 cm of clearance, roll 5.8 deg, saturation 57 %, grace 0.19
  on four tilted starts. The gearing is set by the servo's slew, not by
  the mass, so a faster servo spec is the round that can move it.
  Round 7 (`configs/skater_walk_panto7.json`) is that round: a new
  `servo_sec60` gene (seconds per 60 deg, 0.15 the standard hobby servo,
  0.06-0.08 a high-voltage coreless or brushless one, +15 g per servo at
  twice the speed in `hardware.servo_from_params`) locked at 0.075, so the
  leg servos slew at 14 rad/s instead of 7, from walker_panto6. On the
  round-6 walker alone the fast servo cut the saturation from 57 % to 12 %
  with the gait unchanged; evolved on it (`skater_walker_panto7`), the GA
  spent the speed on a quicker, higher step (1.15 Hz, 14 cm steps, 3 cm
  lift) and a longer strut: 2.3 m in 12 s, 16.7 cm strides, 5.4 cm of
  clearance, saturation 19 %, grace 0.22, fitness -0.11 on four tilted
  starts, against round 6's 1.9 m, 4.6 cm, 57 %, 0.19, -0.15. The roll
  grew to 6.4 deg and the servo acceleration doubled, which is what holds
  the grace below the 6 cm horn's 0.30: the fast servo buys the stride and
  the clearance, not the smoothness. Open: whether the roll gains want
  re-searching for the faster leg, and the fast servo's real speed under
  load (the model slews at the no-load figure).
  Round 8 (`configs/skater_walk_panto8.json`) re-searched the roll gains:
  360 random samples of the roll / lateral balance genes on the round-7
  walker (body and stride fixed, four tilted starts) found the carried-over
  gains already at a local optimum (the best gained 0.01 of fitness, inside
  the noise), and every sample that cut the roll below 4 deg did it by
  shortening the stride and dragging the feet. The GA from the best sample
  with the leg geometry locked (`skater_walker_panto8`) traded a little of
  the stride for a little of the roll: 15.1 cm strides at 1.27 Hz, 4.1 cm of
  clearance, roll 5.6 deg (from 6.4), fans almost off (1.3 % duty), grace
  0.24, fitness -0.10, the same 2.3 m in 12 s. The roll gains are not the
  lever on this leg: the 25 mm pantograph on a 10 cm hip rolls 5.5-6.5 deg
  whatever the gains, and the remaining smoothness gap to the 6 cm horn's
  round 3 (grace 0.30) is in the servo acceleration, not the roll.
  Round 9 (`configs/skater_walk_panto9.json`) made `stance_w` signed (feet
  inboard of the hip points, clamped a sole's width plus a centimetre apart;
  the body shift sized to the stance foot where it actually is) and freed the
  hip width, the user's idea of a wide hip for skating with the body stepping
  on top of its feet for the walk. The round-8 gains do not carry over to
  inboard feet (the walker falls in 2-7 s on either hip), so a random search
  on 16, 20 and 24 cm hips seeded it (`skater_walker_panto9_seed`, 24 cm hip,
  feet 6 cm in, roll 3.6 deg, two of 200 samples upright). The GA
  (`skater_walker_panto9`) went to a 31 cm hip with the feet 5.4 cm inboard
  (track 23 cm): roll 2.6 deg RMS (from 5.6), grace 0.43 (the best of any
  round, the 6 cm horn's 0.30 included), servos never at the slew limit,
  servo acceleration halved, fans 2.3 %, 2.1 m in 12 s, four of four tilted
  starts, fitness -0.09 against -0.10. The cost is the gait: 10 cm strides
  at 1.76 Hz with the swing clearance at zero, a shuffle, which is what the
  roll-free body lets the GA get away with. The roll lever on that hip is
  5.8 horn lengths (roll range 22 deg, roll slew 2.4 rad/s), and the leg
  rolls 17 deg to put the foot where it stands, so the roll servo is now the
  loaded one (the plan's roll-servo spec). Next is `skater_test_plan.md`,
  revised after a controls review: foot sensing and a real actuator model
  first, the controller rewritten around them second, the body third, the
  other modes last.
- **Foot sensors (plan phase 0).** `skater/sensors.py`: two force-sensing
  resistors per foot (heel and toe) modelled from the floor's contact forces
  on the foot geoms, split between the pads by the lever, through an FSR
  response (nothing under 1 % of body weight, soft saturation at 1.5), an
  8 ms low-pass, a 10-bit ADC over two body weights and a dropped reading
  about once per 5 s per channel; per foot the controller reads `load`,
  a debounced `contact` and the centre of pressure `cop`, plus the passive
  ankle's angle and each leg servo's load as a fraction of stall. Read every
  control step, not yet used by the gait, so every score is unchanged to
  the third decimal. The rollout reports `sensor_stance_frac`,
  `touchdown_load_bw` (mean and max), `cop_excursion_m`, `sensor_dropouts`
  and `servo_load_peak`. First readings: the round-8 walk lands at 1.6 body
  weights on average and 2.3 at worst (the landing mode of plan phase 1 is
  needed), the round-9 shuffle at 0.5; and on round 9 **both roll servos sit
  at stall 52 % of the time** (mean 66 % of stall torque; 3 % on the 10 cm
  hip), so the wide-hip walk is carried by servos at their torque limit and
  its 2.6 deg roll is not a build result until the roll servo has its own,
  stouter spec (phase 0) and the walk is re-scored with it.
- **Roll servo spec (plan phase 0).** `roll_servo_kgcm` and
  `roll_servo_sec60` are genes (10-80 kg cm, 0.06-0.20 s per 60 deg), used
  for the roll actuator's torque and stiffness, the roll channel's slew and
  the hip's mass (`hardware.roll_servo_from_params`; every archetype was
  given its leg servo's values, so no gait changed). The scored saturation
  keeps measuring the roll channel against the leg servo's slew as it
  always has; the roll servo's own saturation is reported as
  `roll_servo_sat_frac`. Re-scoring round 9 with a stouter roll servo did
  not take it off its limit: 53 % of the time at stall with 40 kg cm, 37 %
  with 80, the walk otherwise unchanged (fitness -0.06 to -0.09, grace
  0.42-0.43, roll 2.6-2.7 deg). The probe says why: the roll joint sits at
  8.4 deg while the command is 14.4 deg and the joint is never on its stop,
  so the two roll servos are squeezing the planted feet toward each other
  against the floor's friction. The GA's "feet 5.4 cm inboard" is a command
  the feet never reach (the measured track is 23 cm, the commanded 15); the
  2.6 deg roll is bought by preloading the stance through the floor with
  two servos at stall. That is a foot-placement rule set at swing and held
  from the measured foot in stance, which is phase 1's capture-point step,
  not a servo size.
- **Real servo model (plan phase 0).** A task switch, `"servo_model":
  "real"` (default `ideal`, so every score so far stands): the six leg
  servos slew at (1 - load / stall) of their no-load speed (a straight
  speed-torque line with a 5 % floor), receive the command issued 20 ms
  earlier, refreshed at 50 Hz and held between refreshes, and move only for
  a change past a 0.3 deg pulse deadband (`controller._servo_link`,
  constants in `hardware.py`); fans and treads pass straight through.
  Re-scoring the walkers under it on four tilted starts:

  | walker | ideal: upright, fitness, grace | real: upright, fitness, grace |
  |---|---|---|
  | hip3 (four-bar, 6 cm horn, slow servo) | 12 s, -0.08, 0.35 | 12 s, -0.32, 0.35 (1.3 m, not 2.0) |
  | panto7 (25 mm horn, fast servo) | 12 s, -0.12, 0.22 | falls at 2.3 s |
  | panto8 (roll gains re-evolved) | 12 s, -0.10, 0.24 | falls at 2.5 s |
  | panto9 (31 cm hip, feet inboard) | 12 s, -0.09, 0.43 | 1 of 4 starts upright, falls at 5.8 s |

  Every pantograph gait since the horn cap was tuned to a servo that does
  not exist: 20 ms of latency at 50 Hz with the slew falling under load is
  enough to drop them in two or three steps, with the servo acceleration
  tripled on the way down. The four-bar walker, evolved slow and with its
  servos at a quarter of stall, survives at two thirds of its speed. This is
  the controls review's point in numbers, and the reason phase 1 rewrites
  the controller for these servos before any more body rounds: from here
  every round scores under `servo_model: real`.
  Open: the lower links' fore-aft
  reach is the horn's (a 3.5 cm horn cannot move the X 6 cm at a deep
  crouch, so the buildability check rejects it), the leg mass sits at 15 %
  of the body with 12 cm rods (the leg-mass penalty will push on rod length),
  and `tools/joint_trace.py` and `tools/skater_layout.py` still label the
  `hip_pitch` and `knee` joints as thigh and knee: on the pantograph they are
  the front and rear horns.
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


  Phase 0 closed with the result's last three readings and the static check.
  `fall_dir` names the fall (forward, back, left, right, drop). Per-servo
  peaks per gait cycle: on the ideal servo the pantograph walkers' horns peak
  at 0.39-0.40 of stall in the median cycle and 0.62-0.67 in the worst; under
  the real servo the worst cycle is the one they fall in, at stall (1.0), with
  the horn moving at or past its no-load speed (1.0-1.1) as the body drops.
  The four-bar walker's knee hinge reads 1.3-1.9 of the servo's no-load speed:
  the knee is driven through the shin four-bar, so the hinge outruns the horn
  by the linkage ratio, and the landing kicks it. Round 9's roll servos read
  1.0 of stall standing still (the squeeze). The foot track check settles the
  rolled-plane IK question: the four-bar standing lands its ankles within a
  millimetre laterally of the controller's target, so the roll-then-planar
  solve is exact and nothing was changed; round 9's inboard feet sit 12-13 mm
  short standing and about 20 mm walking because the planted soles do not
  slide. The static check (`skater/feasibility.py`) passes every pantograph
  since the horn cap (lift 5.5 cm, fore 10.5, aft 7.5, leg servo at 0.33-0.36
  of stall one-legged, roll servo 0.09-0.19) and fails every four-bar body on
  reach at the crouch height: 3 cm fore (the shin drive's fold margin) and
  5 cm aft (the knee's 10 deg stop). The four-bar's strides were the body
  riding over a foot clamped at its stops. Phase 1 starts from here, on the
  round-9 body, under `servo_model: real`.

  Phase 1, item 1 (2026-11-08): the contact-driven gait phase, behind task
  `gait_phase: contact`. What it took to make it not fall: a landing is a
  quarter body weight for two steps on a foot that has been off the floor
  for three (a brush mid-swing, which the 2-4 cm lifts give every step, had
  counted as a landing and started the stance behind the hip); the stance
  foot's sweep keeps the clock's speed through a phase pull or hold (pulling
  the shared phase to one foot's touchdown had jerked the other foot's
  sweep 4 cm in 50 ms); the swing leaves from where the foot measured at
  lift-off, not from the anchor minus the step. With those, the round-8
  walker runs 12 s under the contact gait on three of four starts with
  every touchdown resolved late (its clock gait lands the foot by shifting
  weight onto it, not by the swing). The gains round on the round-9 body
  under the real servo (`skater_walk_phase1a`) gives a survivor: 12 s
  upright on all four flat starts (round 9: 6 s), grace 0.33, roll 3 deg,
  fans 1 %, servo saturation 2 %; strides 2.7 cm, lift 2 cm, 36 cm in 12 s.
  Twenty early landings per run: the feet find the floor before the clock
  says so and the phase follows them, which is the item working; the body
  barely travels, which is the score. The same gains also stand 12 s on the
  clock. Cobblestone: 7 s mean, one of four finishes. Phase 1 continues
  with capture-point placement and gravity compensation; the robustness
  score (item 5) needs a stride floor or the real servo will keep choosing
  the shuffle.

  Phase 1, item 2 (2026-11-08): capture-point placement, `foot_placement:
  capture`. The first version moved both planted feet to the full capture
  point (velocity x sqrt(h/g) plus the lean h x tilt) and fell in a second
  on every walker, clock and ideal servo included, with 30-50 deg of roll:
  the lean term is 43 cm of foot shift per radian of body roll, through roll
  servos, against planted soles, where the evolved gains had settled at
  0.5 cm per radian. The planted feet's shift is an ankle strategy and
  belongs small; the capture point is a step strategy and belongs to the
  swing foot. The version kept does that: swing landing at velocity x
  sqrt(h/g), weighted over the swing, the kp / kd shift untouched, the kv
  velocity genes dropped. Ablation on the clock gaits: swing-only capture
  keeps the four-bar walker up on four of four starts and the round-8
  walker on three; dropping the kv genes alone costs the round-8 walker two
  starts; sizing the body shift to the measured feet costs two more, and was
  dropped. The gains round on the round-9 body found nothing better than
  its seed: at 3 cm/s the capture offset is 6 mm. The stride floor and the
  capture point both say the same thing from different sides: the real
  servo is the wall, and the next piece of work is finding out which servo
  stalls in the cycle a strider falls in, and whether gravity compensation
  or a different stride shape takes it off stall.

  Stall diagnosis (2026-11-08), before phase 1 item 3. A per-step trace of
  all six servos through the falling cycles of the round-9 and round-8
  striders under the real servo overturns the phase-0 reading. No horn
  stalls in a fall: the hip-pitch and knee servos peak at 0.3 to 0.8 of
  stall. On the round-9 body both hip roll servos are at stall for 93 % of
  every run, standing included, under the ideal servo too (54 %), 3.5 deg
  short of their command, each pushing with the full 22.8 N m and the two
  in opposite directions. It is not floor friction (a frictionless floor
  changes nothing) and not weight (the static check's 9 % of stall is
  right): it is the roll lever's toggle. The horn is horizontal with a
  30 mm vertical link to the bracket end, so the bracket end can only rise
  or drop by a horn length before the horn points straight down and the
  loop stops closing: asin(25 / 144) = 8.6 deg of hip roll, and the
  simulation stops at 8.63 with the horn at 67 deg and the link at 86. The
  range formula (horn sweep over the lever ratio, 22 deg) assumed the
  linear ratio held to 130 deg of horn. `structure.roll_lever_range` now
  solves the loop and the MJCF joint range, the controller's clamp and the
  static check use it; the static check fails the round-9 body (11.2 deg
  needed, 8.5 available), which is to say round 9's inboard-feet stance was
  never reachable and its 2.6 deg roll was the body hanging on two servos
  at stall. The clamp changes the round-9 scores by nothing worth a number
  (the servos could not get there) and takes the roll servos to 0.3 to 0.6
  of stall. Then the real servo model taken apart, one element removed at a
  time, two striders, four tilted starts each. Round-9 strider: the 20 ms
  latency is the whole fall; without it every variant stands 12 s on all
  four starts, and latency alone falls on all four. Round-8 strider:
  latency alone is harmless (12 s on four of four) and the 50 Hz hold with
  the speed-torque line is the fall; its horns at 0.5 to 0.8 of stall in
  the swing slew at half speed, the leg lands late, and the body bounces
  with both feet off the floor in stance. The speed-torque line alone costs
  round 9 four centimetres of stride. Phase 1 item 3 is therefore latency
  compensation (command for when the command lands: gait phase advanced by
  the delay, balance led by tilt rate x delay) before gravity compensation,
  whose real job is the swing horn's load, not the stance's.

  Phase 1, item 3a (2026-11-08): latency compensation, task `servo_lead`.
  The controller commands for when the command lands: the gait clock (or
  contact phase) is evaluated the lead ahead, and roll and pitch are led by
  their rates in every balance term, feet, fans and levelling alike. On the
  round-9 strider under the real servo this is the whole difference: no
  lead, 5.3 s mean with one of four tilted starts standing; 40 ms (the
  20 ms latency plus the 20 ms hold), all four stand 12 s with 7.3 cm
  strides and 1.54 m travelled, fitness -0.35 against -0.49. 30 ms stands
  three, 50 ms one: the lead is the delay, measured, not a gain. Under the
  ideal servo the same lead costs grace 0.02 and a third of the distance,
  which is what predicting a delay that is not there should cost, so the
  default stays 0 and the switch is `auto` under `servo_model real`. The
  round-8 and round-7 striders do not move at any lead (2 to 3 s): the
  ablation had already said their fall is the 50 Hz hold with the
  speed-torque line, the swing horns at 0.5 to 0.8 of stall slewing at half
  speed, and that is item 3b, load compensation on the swing horn. The
  phase-1a shuffler neither gains nor loses up to 30 ms. The round-9 body
  now has a strider that stands under the real servo, which no gains round
  had produced; the next gains round should run with `servo_lead auto`.

  Phase 1, item 3b (2026-11-08): the workspace clamp, task `leg_reach:
  clamp`, in place of gravity compensation, which turned out to have
  nothing to compensate. Hung still in the air the leg reads 0.00 of stall
  on every horn, and in stance the horns read 0.03 to 0.10: the pantograph's
  horns are not loaded by weight. The swing-horn "load" the ablation
  blamed was the position gain meeting a staircase of commands, and the
  staircase was the IK's. A trace of every `panto_ik` call on the round-8
  strider shows its stance leg 1 to 2 cm past the pantograph's 31 cm
  straight-down reach for a third of every stance and 4 cm past it for the
  first 0.4 s, under the ideal servo too: the GA had learned to lean on the
  IK's nearest-pose fallback, which returns the closest pose the damped
  Newton found with 15 to 50 mm of residual, and the stiff ideal servo
  tracked that, which is where the ideal servo's 0.19 to 0.26 of slew
  saturation came from. Under the real servo the fallback wandered onto
  the other solution branch, both horns at +130 deg, and stayed there for
  the whole run. The clamp pulls the ankle target radially toward the hip
  point to a centimetre inside the reach before the IK, and an IK that does
  not converge from the last pose is solved again from the crouch. Under
  the real servo with the lead, four tilted starts: the round-7 strider
  stands 12 s on all four with 14.6 cm strides, 4.7 cm clearance, 2.0 m in
  12 s and fitness -0.35 (from -0.72 and 1.4 s); the round-8 strider three
  of four, 12 cm strides, 1.6 m. The clamp without the lead gives three of
  four on each; the lead without the clamp gives nothing, which is why item
  3a looked like it had missed them. Under the ideal servo the clamp costs
  0.00 to 0.02 of fitness and removes the saturation. The round-9 body and
  the shuffler never leave the workspace and are unchanged. Two things
  follow. The static check's 2 cm of extension headroom is a real
  constraint the controller now honours, and the GA will have to pick a
  deeper crouch or a longer leg if it wants a longer stance. And the
  pantograph's strider is the round-7 body (10 cm hip, 25 mm horn, fast
  servo), not round 9: round 9's inboard stance is beyond its roll lever
  and its best walk under the real servo is a 7 cm stride, while round 7
  now does 15 cm with the body it always had.

  Round 1d (2026-11-08): the first gains round on the round-7 body under
  the real servo with the lead and the clamp. Nothing kept. The GA's best
  (-0.15 on its own two repeats, seed -0.35) stands three of four tilted
  starts and scores -0.44 where the seed stands four and scores -0.35; it
  is also worse under the shove. It is round 1c's result from the other
  side: a gains round over the mean of two starts finds a knife edge, and
  the next thing to build is the robustness score (item 5), not another
  round. Two things the round turned up: the round-7 stand under the real
  servo falls on one or two of four shoves (never tuned for it), and the
  contact gait and the capture point, both tuned on the round-9 shuffler,
  drop the round-7 strider at the start and need their own round on it.
  The tread drive box also moved today, to the shin's outer face halfway
  between its pins, after a clearance check found the old inner-face
  placement 3 mm into the front rod's extension at a 5 cm lift; round 7's
  scores did not change for it.

  Phase 1, item 5 (2026-11-08): the robustness score. Four task keys,
  default off: `aggregate` (p20 or min over the repeats instead of the
  mean), `mass_error` (the hip's mass and inertia scaled by a uniform
  fraction per run), `friction_error` (every friction coefficient
  likewise) and `push` (one shove at a random moment of a legged run, a
  fraction of the stand's). The errors are drawn after the controller has
  read the nominal model, which is the point: the controller is wrong by
  that much. One wrong turn on the way, a constants refresh after scaling
  the mass that dropped every robot at t = 0; the dynamics read the mass
  directly and the refresh was not wanted. The score measured on the three
  walkers in hand says what the plan said it would: the round-1d GA best,
  -0.15 on its own two repeats, is -1.32 at the 20th percentile of eight
  perturbed starts; the round-7 strider is -0.86, striding seven of eight
  flat starts for 12 s and falling on the eighth and on one stand; and the
  round-9 shuffler is -0.39, the best of the three, because it never falls
  and stands the shove well. That is the score rewarding safe non-behaviour
  again, now with numbers, and it is why the robustness round carries the
  stride floor at 10 cm: a 2 cm shuffle is speed-gated to a fifth and pays
  the stand cost. Round 1e ran on the round-7 body with that score.

  Round 1e (2026-11-08): the first gains round that kept its winner. GA
  best -0.37 against the seed's -0.86, and, which is the point of item 5,
  the gap holds on starts the GA never saw: -0.74 against -1.16 at the
  20th percentile, eight of eight perturbed flat walks standing where the
  seed stood five, four of four perturbed stands where the seed stood two.
  The GA bought it with balance gain (`kp_bal` 3.2 to 5.3), fan pitch and
  yaw gains, and foot velocity feedback that the seed had at zero, and it
  paid in stride: 12 cm against 15, clearance 3.0 cm against 4.3, which is
  the stride floor doing exactly what it was set to do and no more. Fans
  went from 5.5 % to 1.7 % of duty with `fan_base` at zero, so the fans
  now fire only on demand. Cobbles drop both walkers inside three
  seconds; the rough task is the next unsolved thing on this body, and
  the contact gait and the capture point (items 1 and 2) still have to
  be re-tuned on it under this score.

  Rounds 1f and 1g (2026-11-08): items 1 and 2 on the strider. The capture
  point round kept its promise and broke a rule at the same time. Its
  winner beats the seed on unseen seeds (-0.58 to -0.83 under capture, and
  to -0.74 under the seed's own clock gait), stands seven of eight
  perturbed flat walks and every stand, rolls 5.3 deg RMS where nothing
  before rolled under 6, and runs the fans 1.5 % of the time. It also
  drags its feet: 1.0 cm of swing clearance against the seed's 3.0 and the
  plan's 3. The score has a stride floor and no clearance floor, and the
  GA found the gap in one round, as it finds every gap. The next capture
  round carries a clearance floor. The contact gait round found nothing,
  and the probe before it said why: the strider's clock gait is a 91 ms
  commanded swing that the real servo smears into a dynamic swap, the
  body is never balanced on one foot, and a gait that waits for a
  landing has nowhere to wait. Longer swings, longer waits and a paused
  stance all fell inside two seconds. That is a controller problem for
  item 1 on this body, not a tuning problem, and the GA's answer (improve
  the stand, fall at 0.9 s anyway) is what a flat fitness landscape looks
  like from the inside. A new tool out of the session: tools/live_ga.py
  watches a run live, with the champion of each generation filmed.

  Item 1 on the strider (2026-11-08): the controller change, found by
  switching the contact gait's parts on one at a time rather than by
  designing. With every part off and an instant start the contact gait is
  the clock gait and walks like it. The parked start kills the strider on
  its own: it cannot stand still on split feet, it is a dynamic walker
  whose static balance gains the GA inverted long ago. The measured
  anchors kill it on their own: under the real servo its foot lands 4 cm
  behind its command, and a stance started from where the foot is, rather
  than where it was asked to be, walks the support out from under the
  body step by step. The late hold, the part that reads the sensors, is
  the part that helps. The grand version, a true single-support gait with
  the hip carried over the stance foot, fell in three quarters of a second
  every time, and the mechanism is worth remembering: on this body the
  lateral shift goes through the hip roll servos, which roll the body as
  much as they move it, and the roll-levelling term then shortens the leg
  the body is moving onto and lifts it off the floor. Single support on
  this body wants an ankle, or a wider sole, or a hip that translates;
  that is phase 2. Item 1 lands as the weak form: landing-gated timing
  behind contact_start 'instant' and contact_anchor 'command', with the
  shuffler's defaults untouched. The clearance floor (clear_min) went in
  at the same time, so round 1h cannot win the way round 1f did.

  Round 1h (2026-11-08) won the way the floors allow: -0.91 against the
  seed's -1.10 on unseen seeds, every plain start and every perturbed
  stand upright, 4.7 cm of clearance, and 10.5 cm strides sitting exactly
  on the 10 cm floor. The same count of perturbed flat falls as the seed
  (two of eight, now backward rather than sideways). Kept as the item-1
  walker because it is the first sensor-gated gait that strides, and
  named for what it is: a modest, honest gain, not a new walker. The
  pattern across rounds 1e to 1h is now clear enough to act on. Each
  gains round buys stability with stride until it meets a floor, and the
  GA's own best (-0.29) is never the unseen-seed number (-0.70 on the
  flat task): a population of 96 over 40 generations overfits its eight
  starts. That is the case for the population-shape test and for CMA-ES
  (item 6), and it is the case for stopping the gains rounds on this body
  until the score's starts are drawn fresh each generation.

  The population-shape test (2026-11-08) asked whether round 1e's 96 x 40
  was the right way to spend 3840 evaluations. Wide and short (192 x 20)
  found its best at generation 6 and scored -0.91 on unseen seeds. Long and
  narrow (48 x 80) kept climbing to a GA best of -0.25, the best number any
  round has shown, and scored -0.71 on unseen seeds against round 1e's
  -0.74. So the long run learned its eight starts far better and the walk
  in general only a little better. What it did learn is the kind the user
  asks for: the steadiest walk yet (roll 5.1 deg RMS, servos saturated 5 %
  of the time), slightly longer strides, but feet lifted only 2.1 cm and one
  perturbed stand lost. Kept as a seed. The shape question is answered
  (long and narrow), and the answer matters less than the overfitting it
  exposed: the next change is fresh starts each generation.
