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
   item 10.
2. **Capture-point stepping.** The lateral foot placement from the linear
   inverted pendulum (foot to the capture point, `y + v_y sqrt(h/g)`),
   replacing `kp_roll_y`, `kd_roll_y` and `kv_foot_y`; the fore-aft
   placement likewise, replacing the `kp_foot_leg` family. Correct at any
   hip width by construction, which is what broke between rounds 8 and 9.
   The body shift sized to the measured stance foot, not the gene.
3. **Gravity compensation on the horns.** The leg is a known linkage: the
   horn torques that hold the body's weight at the current pose are
   computable and fed forward, so the servos' position loops carry only the
   error. Removes most of the "shift per rad of pitch" gains.
4. **Landing mode.** On a contact rise after a flight (or after a swing on
   rough ground) the leg's stiffness drops for a few hundred ms through the
   horn command, then recovers. Needed before the run and the hop.
5. **Robustness as the score.** From here every round scores the worst or
   the 20th percentile over 8-12 random starts (tilt, push, a 10 % mass
   error and a 20 % friction error), not the mean of two; the winner is
   re-run with a dropped sensor reading every few seconds and must survive
   it. The knife-edged gaits in memory are the mean-of-two signature.
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
controllers fall in the blend.

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
