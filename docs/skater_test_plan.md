# Skater test plan: sense the feet, fix the controller, then the best traveller

Written 2026-11-07 after walking rounds 1-9 on the pantograph leg, revised
the same day after a controls review. The first draft went straight from the
walking rounds to "free the body and find the best walker, then the other
modes". The review said the body is not the bottleneck: every round since
the horn cap found a controller limit (a servo slew limit, a limit cycle in
the balance gains, a roll the gains cannot remove, a shuffle the score lets
through), and the controller is a hand-written PD cascade with ~25 gain
genes that a GA tunes blind, driving a clock gait on servos modelled at
their no-load speed with no contact sensing at all. So the plan now runs:
sensors and actuator model first, controller second, body third, modes last.

Every round keeps the standing rules: scored on several tilted starts, not
the GA's own number; the clip is watched; grace, stability and few fans
count over speed; long strides over cadence; feet picked up; fall direction
reported. A round is a config under `configs/`, an archetype under
`configs/archetypes/`, a paragraph in `skater_review.md` and a commit.

## Phase 0: what the robot can feel and what the servos can really do

Hardware the real build will have, modelled before the controller is written
to use it. Nothing here needs a GA round; each item is a commit with a test.

### Foot sensing

- [x] **Two force-sensing resistors per foot** (`skater/sensors.py`), heel and toe, under the sole
      plate. In the simulator: the floor contact force on the sole summed
      into two bins by where it lands along the sole, low-pass filtered,
      quantised, with an FSR-like response (zero below a threshold, soft
      saturation above the body weight) and a dropped reading now and then
      so the controller never sees cleaner data than the real sensor. The
      controller gets, per foot: `load` (heel + toe, in body weights),
      `contact` (load above a debounced threshold) and `cop` (the centre of
      pressure along the sole, from the heel / toe split).
- [x] **Ankle angle** from the passive ankle's deflection (a hall sensor and
      magnet across the joint on the build; the joint angle in the model),
      as a second, independent load estimate and the landing-stiffness
      input.
- [x] **Servo current** as a load estimate: the model's actuator force as
      a fraction of stall, filtered with the servo's electrical lag, per
      leg servo. Reported, not yet used.
- [x] The result dict reports, per step: load at touchdown (the impact), the
      centre of pressure's excursion along the sole, and the stance
      fraction measured from the sensor rather than the clock.

### Actuator model

- [x] **Speed under load** (task `servo_model: real`): the servo's slew falls linearly from the no-load
      figure to zero at stall, so a fast servo working hard is not fast.
- [x] **Deadband, latency and update rate** (same switch): a 4 deg deadband (already the
      stiffness model), a 20 ms command latency and a 50 Hz command rate so
      the derivative gains are tuned against the servo the build will have,
      not a continuous one. Expect the round-5 limit cycle to move.
- [x] **A roll servo spec of its own** (`roll_servo_kgcm`,
      `roll_servo_sec60`): on the wide hip the roll servo carries the leg
      through a 5-6x lever and is the stout one; on the 10 cm hip it idled
      at 3 % of its torque.
- [x] **Fall direction** in the result (`fall_dir`: forward, back, left,
      right, from the body's pitch and roll at the fall; `drop` when the hip
      sank with neither past 10 deg).
- [x] **Per-servo peak torque and speed per step** in the result
      (`peak_torque_hip_roll` / `hip_pitch` / `knee` and `peak_speed_*`: the
      median gait cycle's peak, fraction of stall and of the no-load speed at
      the hinge, the worse side; `servo_peak_torque_frac` / `speed_frac` the
      worst cycle of all six), since those decide whether the servos survive
      the gait. First reading: the pantograph walkers run their horns at
      0.6-0.65 of stall in the worst cycle on the ideal servo and hit stall
      (1.0) in the cycle they fall under the real one; round 9's roll servos
      sit at 1.0 standing still.

### Kinematics

- [x] **IK in the rolled leg plane.** Checked rather than changed: the roll
      axis is the body's x, so a fore-aft step stays in the rolled plane and
      the controller's roll-then-planar solve is exact (the four-bar standing
      lands its ankles within a millimetre laterally of the target). The
      controller now keeps the target (`foot_target`, hip frame) and the
      result reports the ankle's RMS distance from it (`foot_track_rms_m`,
      `foot_track_y_rms_m`). Round 9's inboard feet sit 12-13 mm short of
      their lateral target standing and 18-21 mm walking: the planted soles
      will not slide, which is the roll-servo squeeze again, not the solve.
- [x] **The roll lever's range** (`structure.roll_lever_range`, 2026-11-08,
      found by the phase-1 stall diagnosis): the drive's horn is horizontal
      with a vertical link to the bracket end, so the hip roll stops where
      the link toggles, about asin(horn / arm): 8.5 deg on the round-9 body,
      where the sweep / ratio formula gave 22. MJCF range, controller clamp
      and static check use it; round 9's inboard stance is infeasible by it.
- [x] **Static feasibility check** (`skater/feasibility.py`, no simulation):
      lift, fore, aft and extension reach from the crouch; the leg servos'
      torque to hold the body on one leg at the worst of those poses (with a
      30 % fore-aft shear) and the roll servo's to hold it over the standing
      foot, as fractions of stall, under 0.8; the hip roll the standing track
      needs against the roll range; the soles' gap at that track. Every
      pantograph since the horn cap passes (lift 5.5 cm, fore 10.5, aft 7.5,
      leg servo 0.33-0.36 of stall). Every four-bar body fails on reach: at
      the crouch height the shin drive folds at 3 cm fore and the knee's 10
      deg stop at 5 cm aft, so the four-bar's 12-16 cm strides are the body
      riding over a foot the controller has clamped at the stops. The body
      rounds of phase 2 sample only bodies that pass it.

## Phase 1: the controller, written for the sensors

The gait and balance rewritten so the GA tunes a few physical parameters
instead of 25 blind gains. Each item is a controller change plus a gains
round on the fixed round-9 body (31 cm hip, feet 5.4 cm inboard, fast
servo), scored against round 9 on the same four tilted starts.

1. **Contact-driven gait phase.** Stance begins on the foot sensor's
   contact rise, not on the clock; the swing ends when the foot finds the
   floor, so uneven ground stops the leg early instead of late. The clock
   stays as a fallback when a foot never lands. Start and stop become a
   sequence on the sensors: shift, confirm the load on the stance foot,
   lift the other. Measured on flat and cobblestone, the tilted starts and
   a stand-walk-stand profile.
   *Done 2026-11-08 for the walk* (task `gait_phase: contact`; the clock is
   the default and every old score stands): the shared phase runs at
   `step_hz`, a swing foot that reads a quarter body weight for two steps
   after having been off the floor pulls the phase to its touchdown over
   50 ms, a foot that has not found the floor by its mark holds the phase
   for 100 ms while searching 3 cm down, then the clock takes over; each
   stance sweep starts from the foot's measured landing point in x and
   keeps its lateral landing error, so a planted foot is never asked to
   slide; the stance sweep runs at the clock's speed through every phase
   pull or hold; the start parks at the left swing until the right FSRs
   read 0.6 body weight. Dropped onto the clock-tuned gaits it fell in two
   steps until a mid-swing brush of the floor stopped counting as a landing
   and the stance anchor stopped absorbing the swing foot's path. Round
   (`configs/skater_walk_phase1a.json`, body locked, real servo): the
   winner `skater_walker_phase1a` stands 12 s on flat where round 9 fell at
   6 s, grace 0.33, roll 3 deg, fans 1 %, but on 4 cm steps with 2 cm of
   lift at 1.1 Hz: 36 cm in 12 s, a shuffle (and it also stands 12 s on the
   clock with the same gains). The real servo plus this score reward
   survival over travel; items 2, 3 and 5 are what make a real step
   affordable. The course (stand, walk, stand) stays on the clock until
   item 10. A stride floor was tried next (task `stride_min`, kept, default
   off: the speed gate also scales with stride / floor): it moves nothing
   here, because the shuffler is already fully gated by its speed (3 cm/s
   against a 15 cm/s gate) and still beats every strider, each of which
   falls. Under the real servo the score has no gradient toward walking
   until something walks; that is items 2 and 3, not the score.
2. **Capture-point stepping.** The lateral foot placement from the linear
   inverted pendulum (foot to the capture point, `y + v_y sqrt(h/g)`),
   replacing `kp_roll_y`, `kd_roll_y` and `kv_foot_y`; the fore-aft
   placement likewise, replacing the `kp_foot_leg` family. Correct at any
   hip width by construction, which is what broke between rounds 8 and 9.
   The body shift sized to the measured stance foot, not the gene.
   *Done 2026-11-08, narrower than written* (task `foot_placement:
   capture`; default `gains` keeps every score). The full capture point
   applied to both planted feet, replacing the kp / kd gains too, fell in
   a second on every walker: it is a 40 cm per radian body translation
   through the roll servos against feet that do not slide. What stands is
   the two strategies kept apart: the planted feet keep the kp / kd shift
   (the ankle strategy, a centre-of-pressure shift within the soles, which
   the GA has always tuned small) and only the swing foot lands at the
   capture point, hip velocity x sqrt(h / g) ahead of its nominal landing,
   weighted in over the swing; the `kv_foot_x` / `kv_foot_y` velocity genes
   are ignored in this mode. Sizing the body shift to the measured feet
   unsettled every clock-tuned gait and was dropped. Round
   (`configs/skater_walk_phase1c.json`, from the phase-1a walker): forty
   generations found nothing better than the seed, whose gains under the
   capture mode stand 10.8 s on flat against 12 with the velocity genes.
   The reason is arithmetic: at 3 cm/s the capture offset is 6 mm, and so
   was the velocity term it replaced. Capture-point placement earns its
   keep at walking speed, and nothing walks at walking speed under the real
   servo yet. Keep the mode; it is right once something strides.
   *Diagnosis before item 3 (2026-11-08): what the real servo actually
   does to a strider.* The belief carried from phase 0, that a horn is at
   stall in the cycle a strider falls in, was a misread of the worst-cycle
   torque metric. A per-step trace of all six servos through the falling
   cycles of the round-9 and round-8 striders shows the horns peak at 0.3
   to 0.8 of stall and never stall in a fall; what stalls on the round-9
   body is both hip ROLL servos, at 93 % of all steps, standing included,
   and under the ideal servo too (54 %). That is a linkage limit, not a
   load: the roll drive's horn is horizontal with a vertical link to the
   bracket end, so the bracket end can only rise or drop by a horn length
   before the link toggles, asin(horn / arm), 8.6 deg on the 144 mm arm and
   25 mm horn, where `hip_roll_range` (horn sweep / lever ratio) said 22.
   The controller asked 12 to 14 deg of roll for the inboard stance and the
   servos pushed against the toggle at full torque for the whole run. Fixed
   in `structure.roll_lever_range` (the loop solved; the MJCF joint range,
   the controller's clamp and the static check all use it). The static
   check now fails the round-9 body: its stance needs 11.2 deg of roll and
   the lever gives 8.5. With the clamp the round-9 gaits score the same
   (the servos could never get there anyway) with the roll servos at 0.3 to
   0.6 of stall instead of 1.0. Memory's earlier reading, that the two roll
   servos squeeze the planted feet against floor friction, was wrong too: a
   frictionless floor changes nothing. Then the real servo model taken
   apart one element at a time on the two striders, four tilted starts
   each: on the round-9 strider the 20 ms latency alone is the fall (every
   variant without latency stands 12 s on all four starts; latency alone
   falls on all four), and the hold, deadband and speed-torque line do
   little; on the round-8 strider latency alone is harmless and the 50 Hz
   hold together with the speed-torque line is the fall (the horns at 0.5
   to 0.8 of stall during the swing slew at half speed, the leg arrives
   late, the body bounces: both feet leave the floor in stance). The
   speed-torque line also costs stride on its own, 10 cm to 6 on round 9.
   So item 3 is reframed below: latency compensation first, then the
   swing's horn load, and gravity compensation as the second half.
3. **Latency compensation, then gravity compensation on the horns.** The
   real servo's 20 ms latency and 50 Hz hold are a known, fixed delay in a
   100 Hz loop, so the controller commands for when the command lands: the
   gait trajectory evaluated at `t + delay` (the clock or contact phase
   advanced by the delay), and the balance terms led by the delay (tilt +
   delay x tilt rate), which is the Smith-predictor-lite that every servo
   biped runs. Checked first on the round-9 strider, where latency alone
   is the fall. Then the horns: the leg is a known linkage, so the horn
   torques that hold the body's weight (stance) and the leg's own weight
   (swing) at the current pose are computable; the position commands are
   offset by that torque over the horn stiffness so the position loops
   carry only the error, and the swing horn's load falls below the
   speed-torque knee. Removes most of the "shift per rad of pitch" gains.
   *3a done 2026-11-08* (task `servo_lead`: seconds or `auto` = latency
   plus hold, 40 ms; default 0 keeps every score): the gait is evaluated
   `servo_lead` ahead and roll / pitch are led by their rates in every
   balance term. The round-9 strider under the real servo, four tilted
   starts: no lead 5.3 s mean and one of four stands; 20 ms two; 30 ms
   three; 40 ms all four stand 12 s, 7.3 cm strides, 1.54 m, fitness -0.35
   against -0.49; 50 ms one. The lead is the measured delay, not a gain to
   tune: half the hold is too little, the hold plus a frame too much. On
   the ideal servo the lead costs a little (grace 0.42 to 0.40, 1.6 m to
   1.25), as a predictor of a delay that is not there should. The round-8
   and round-7 striders are unmoved at any lead (2-3 s), as the ablation
   said: their fall is the hold with the speed-torque line, the swing horn
   slewing at half speed under load, which is 3b. The phase-1a shuffler is
   unmoved up to 30 ms and loses a start at 40. Cobblestone unmoved
   everywhere (1.5 s): rough ground is the contact gait's job and these are
   clock striders.
   *3b done 2026-11-08, and not as written.* The horns carry no gravity load
   worth compensating: the leg hung still in the air reads 0.00 of stall on
   every horn, and the stance horns read 0.03 to 0.10. What the round-8 and
   round-7 striders' horns were doing was the IK's fault, not the servo's.
   Their stance leg runs 1 to 2 cm past the pantograph's straight-down
   reach (31 cm; the crouch sits 2.3 cm inside it) for a third of every
   stance, and 4 cm past it in the first 0.4 s, even under the ideal servo:
   the GA had learned to lean on `panto_ik`'s nearest-pose fallback, which
   the stiff ideal servo tracked with 15-50 mm of foot error and a slew
   saturation of 0.19-0.26 that was this chatter. Under the real servo the
   fallback's damped Newton wandered onto the other solution branch (both
   horns at +130 deg) and stayed there for the whole run. The fix is the
   workspace clamp (task `leg_reach: clamp`; default `free` keeps every
   score): the ankle target is pulled radially toward the hip point to
   `REACH_MARGIN` (1 cm) inside the reach before the IK, and an IK that
   does not converge from the last pose is solved again from the crouch.
   With the clamp and the lead, under the real servo and four tilted
   starts: round 7 stands 12 s on all four with 14.6 cm strides, 4.7 cm
   clearance, 2.0 m in 12 s, fitness -0.35 (from -0.72 and 1.4 s); round 8
   three of four, 12 cm strides, 1.6 m (from 0.4 m). The clamp alone,
   without the lead, gives three of four on each. Under the ideal servo the
   clamp changes their scores by 0.00 to 0.02 and takes the saturation from
   0.19-0.26 to 0.01-0.04. The round-9 body and the shuffler never leave
   the workspace and do not move. The pantograph now has a strider that
   stands under the real servo with the body it had in round 7, which the
   round-9 body cannot match: its stance is beyond its roll lever.
   *Round 1d (2026-11-08, `configs/skater_walk_phase1d.json`): the round-7
   body locked, every task under the real servo with the lead, the walk
   under the clamp, clock gait.* The contact gait and the capture point both
   drop this gait at the start (0.9 s and 2-5 s), tuned as they were on the
   round-9 shuffler, so they stayed off. Forty generations, GA best -0.15
   against the seed's -0.35 on the GA's own two repeats; on four tilted
   starts the "best" stands three of four and scores -0.44 against the
   seed's -0.35 on four of four, and stands worse under the shove. Nothing
   kept (the archetype was deleted). Same lesson as round 1c, from the
   other side: the mean of two starts is a knife edge the GA walks along,
   and item 5 (the robustness score) is now the binding item, before
   another gains round. Found on the way: the round-7 stand under the real
   servo falls on one or two of four shoves (its stand gains were never
   tuned under it), and its cobblestone walk lasts 2-4 s.
   *Also 2026-11-08: the tread drive box moved* (`mjcf.TREAD_BOX_X / _Z`)
   from the shin's inner face 30 mm below the top pin, where it ran 3 mm
   into the front rod's extension at a 5 cm lift and 8 mm into link_b at
   the workspace edge (the model never collides link to link, so nothing
   had said so), to the outer face halfway between the two pins, 22 mm off
   the link: 19 mm clear in the walk, 11 mm anywhere in the workspace. The
   user's reasons: crash avoidance, and a buildable model, where the drive
   and a pair of fingers both have to fit beside the four-bar. Round 7
   rescored identically on flat (12 s on four, -0.34), a little better on
   cobblestone and the stand, within the noise.
4. **Landing mode.** On a contact rise after a flight (or after a swing on
   rough ground) the leg's stiffness drops for a few hundred ms through the
   horn command, then recovers. Needed before the run and the hop.
5. **Robustness as the score.** From here every round scores the worst or
   the 20th percentile over 8-12 random starts (tilt, push, a 10 % mass
   error and a 20 % friction error), not the mean of two; the winner is
   re-run with a dropped sensor reading every few seconds and must survive
   it. The knife-edged gaits in memory are the mean-of-two signature.
   *Done 2026-11-08* (task keys, all default-off so every old score and
   every seed's random stream stand): `aggregate` `pNN` / `min` scores a
   task's repeats at that percentile / the worst instead of the mean (the
   result carries `fitness_mean`, `fitness_min` and `upright_min` beside
   it); `mass_error` and `friction_error` scale the hip's mass and inertia
   and every friction coefficient by a uniform fraction per run, drawn
   after the controller has read the nominal model, so the controller is
   wrong by that much; `push` shoves a legged run once, at a random moment,
   by that fraction of the stand's shove. Sensor dropouts were already in
   every run (one per channel per 5 s). The score on the three walkers
   that matter, phase-1e tasks (8 flat / 4 rough / 4 stand starts, 2 deg
   tilt, 10 % mass, 20 % friction, half a shove, p20): the round-1d GA
   "best" -1.32 (it had scored -0.15 on its two repeats: the knife edge,
   measured), the round-7 strider -0.86 (one flat start falls at 2.5 s and
   one stand at 2.1 s; the other seven flat starts stride 12 s), the
   round-9 shuffler -0.39, because it never falls and stands the shove
   best. That last number is the warning the stride floor was built for,
   so the round carries `stride_min` 0.10. Round 1e (`configs/
   skater_walk_phase1e.json`, round-7 body, real servo, lead, clamp) ran
   on this score: GA best -0.37 (seed -0.86). Scored on seeds the GA never
   saw (base seed 7) against the seed: weighted p20 -0.74 vs -1.16; flat
   walks stand 8 of 8 (one at 9.7 s) vs 5 of 8, perturbed stands 4 of 4
   vs 2 of 4, strides 12 cm vs 15 cm, clearance 3.0 vs 4.3 cm, fans 1.7 %
   vs 5.5 %, distance 1.7 m vs 1.6 m; both fall on cobbles inside 3 s.
   Four plain tilted starts: both 12 s, -0.38 each. Kept as
   `skater_walker_phase1e.json`: the first gains round whose winner beats
   its seed on unseen starts. Genes that moved most: `kp_bal` 3.2 -> 5.3,
   fan pitch/yaw gains up, `fan_base` to 0, foot velocity gains from 0,
   `ki_foot_leg` doubled. Rough terrain is unsolved by either walker.
6. **Gains by CMA-ES.** The remaining gain genes tuned on the fixed body by
   CMA-ES over the robustness score, which converges where the GA's best
   has wandered 0.1 between generations 20 and 40 in every round.

Done when: the round-9 body walks the robustness score upright, with
sensor-driven stance, picking its feet up (clearance at or above 3 cm,
strides at or above 12 cm), roll under 4 deg RMS, fans under 2 %, and the
clip looks like a walk.

## Phase 2: the best traveller (walking, body free)

With the controller written, the body question is small enough to answer.

7. **Round A, body free.** From the phase-1 winner. Unlock the hip, core
   height, battery, payload and fan placement, leg lengths, horn stiffness,
   servo torque and speed (leg and roll), the foot (sole length, tread
   width, foot type), tread motor. Keep the fan hardware spec (a catalogue
   choice) and the 25 mm horn cap; sample only bodies that pass the static
   feasibility check. Gains re-tuned by CMA-ES on the top few bodies, since
   a body is only as good as its tuned controller.
8. **Round B, long run.** The same for 120 generations with the mutation
   sigma halved, the elite archive kept.
9. **Round C, robustness.** The round-B winner on 12 tilted starts,
   cobblestone at full weight, a 30 s walk and a stand with shoves. Fix
   what fails as a gains round, not a geometry round.

Done when: the robustness score upright, grace at or above 0.40, roll under
4 deg RMS, fans under 2 %, strides at or above 15 cm, clearance at or above
3 cm, servo saturation under 20 %, no servo past 60 % of stall at touchdown.

## Phase 3: all-mode travel on the phase-2 body

The other objectives exist and have archetypes, all on the old four-bar leg
with a 6 cm horn and a slow servo; none has run on the pantograph. Each gets
a seed search before its round. The body is locked; the mode's gait and gain
genes move. Modes are explicit states with sensor-triggered transitions and
a defined stance at each switch, in place of the `trans_s` blend: blended
controllers fall in the blend. Agreed 2026-11-08: each transition (start,
stop, walk to run and back, the jump's crouch / thrust / tuck / land) is a
precomputed pose-and-time table, checked offline for reach, torque and
balance at every pose by the static check, with sensor-triggered advance
between rows; `trans_s` goes. The walk itself is not a table: its placement
is live (capture point) and its phase is the feet's.

10. **Stand, start, stop.** Profile stand 2 s, walk 6 s, stand 4 s. The
    transition pitch and the stopping distance are the scores already in
    `_course_scores`. The cheapest mode and the one a real build hits first.
11. **Run.** `configs/skater_run4.json` tasks on the pantograph: flight
    fraction, flight length, bob, with the landing mode. The fast servo was
    bought for the walk's slew; the run is where it should pay. Expect the
    pantograph's work space (a 12 cm step at the crouch) to be the limit.
12. **Gait changes.** The `skater_course3.json` profile (stand, walk, run,
    walk, stand) as a state machine on the sensors: the transition pitches
    are the score. If the run of step 11 does not exist, walk, stop, walk.
13. **Jump.** `configs/skater_hop.json` (hop, broad jump). The pantograph's
    two servos share every load, which is the argument for it; the hop and
    the landing are where that shows or does not. Let the servo torque gene
    move, with its mass.
14. **Skate.** `configs/skater.json` (skate flat, cobblestone, slalom,
    stand). Untouched since the leg changed, and a different control
    problem: a wheeled inverted pendulum, not a biped. Snapshot the skate
    stance first (`tools/skater_snapshot.py`): the crouch and the tread's
    position under the body were set for walking. The wide hip is the
    skating stance it was widened for.
15. **The traveller.** One profile with every mode: stand, walk, run, walk,
    stop, skate, stop, hop. Scored as the course is, the skate segment on
    the tread speed. Body locked; a long gains round. The clip is the
    deliverable.

## Order and cost

Phase 0 is done (2026-11-08): code and tests, no GA. Phase 1 is six
controller changes, each with a gains round of 15-20 minutes on 12 workers,
plus the CMA-ES runs. Phase 2 is about 2.5 hours of compute including the
long round. Phase 3 is about 3 hours plus a seed search per mode. Phase 0's
foot sensing, speed under load and rolled-plane IK come before anything
else; round 9 is not believed until the IK check has run.

## What is deliberately not in the plan

- GPU physics or a different simulator (memory: no GPU physics).
- A new leg topology. The pantograph is the leg until phase 3 says otherwise.
- Hip yaw. Turning stays tread differential plus fan yaw.
- Freeing the fan hardware spec: a catalogue choice, locked by group.
- A learned controller. The rewritten controller stays a small, readable
  cascade with physical parameters, so the build can be debugged with a
  multimeter.
