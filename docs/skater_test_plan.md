# Skater test plan: the best traveller, then all-mode travel

Written 2026-11-07, after walking rounds 1-9 on the pantograph leg. The
walking rounds so far have each changed one thing with the rest of the body
locked (see `skater_review.md`, "Pantograph leg"), which answers one question
per round but has fixed the hip, fans and foot at the round-5 seed's values
since the 25 mm horn cap went in. This plan does the opposite in two phases:
first free everything and find the best walker the search can reach, then
take that body through the other modes of travel.

Every round keeps the standing rules: scored on four tilted starts, not the
GA's own number; the clip is watched; grace, stability and few fans count
over speed; long strides over cadence; feet picked up; fall direction
reported. A round is a config under `configs/`, an archetype under
`configs/archetypes/`, a paragraph in `skater_review.md` and a commit.

## Phase 0: the knobs that are not genes yet (one commit, no round)

Things the plan needs that the code does not expose. Each is small.

- [x] `leg_mass_penalty` in the config (round 6).
- [x] `servo_sec60`, leg servo speed, fast servos heavier (round 7).
- [x] `stance_w` signed: feet inboard of the hip points, clamped a sole's
      width plus a centimetre apart; body shift sized to the stance foot
      where it actually is (round 9).
- [ ] A roll servo spec of its own (`roll_servo_kgcm`, `roll_servo_sec60`):
      on a wide hip the roll servo carries the leg through a 4-5x lever and
      is the stout one; on the 10 cm hip it idles at 3 % of its torque.
- [ ] Servo speed under load: the slew is the no-load figure. A speed-torque
      line (speed falls linearly to zero at stall) would make the fast-servo
      result honest.
- [ ] Fall direction in the rollout result (`fall_dir`: forward, back, left,
      right, from the body's pitch and roll at the fall), so a failing round
      says how it failed.
- [ ] The IK in the rolled leg plane: with inboard feet the leg plane is
      tilted and a fore-aft step has a lateral component; check the foot's
      actual track against the target in a clip before trusting round 9.

## Phase 1: the best traveller (walking, everything free)

Goal: the walker with the best grace on four tilted starts, no gene held at a
value chosen by hand. Three rounds, each from the previous winner.

1. **Round A, body free.** From the best of rounds 8 and 9. Unlock the hip,
   core height, battery, payload and fan placement, the leg lengths, the
   horn stiffness, servo torque and speed, the foot (sole length, tread
   width, foot type), tread motor. Keep the fan hardware spec (a catalogue
   choice) and the 25 mm horn cap. Population 96, 40 generations, the walk
   config's three tasks. Expect the leg-mass penalty and the servo mass to
   be the only things stopping the body from growing; watch the hip width
   and the stance sign, which is the wide-hip question put to the GA.
2. **Round B, long run.** The same config for 120 generations with the
   mutation sigma halved, since 40 generations have not been converging
   (the GA's best wandered 0.1 in fitness between generations 20 and 40 in
   every round). Save the elite archive, not just the best.
3. **Round C, robustness.** The round-B winner re-scored on 12 tilted starts,
   cobblestone at full weight, a 30 s walk for long-duration stability, and
   a stand with shoves. Fix whatever fails as a gains round (geometry
   locked) rather than a geometry round.

Done when: four of four tilted starts upright, grace at or above the 6 cm
horn's 0.30, roll under 5 deg RMS, fans under 2 % duty, strides at or above
15 cm, servo saturation under 20 %, and a clip that looks like a walk.

## Phase 2: all-mode travel on the phase-1 body

The other objectives exist and have evolved archetypes, all on the old
four-bar leg with a 6 cm horn and a slow servo; none has been run on the
pantograph. Each gets a seed search (random, the balance and gait genes,
four starts) before its GA round, since every leg change so far has needed
one. The body is locked at the phase-1 winner; only the mode's gait and
gain genes move.

4. **Stand, start, stop.** `course` profile stand 2 s, walk 6 s, stand 4 s.
   The transition pitch and the stopping distance are the scores already in
   `_course_scores`. This is the cheapest mode and the one a real build hits
   first.
5. **Run.** `configs/skater_run4.json` tasks on the pantograph: flight
   fraction, flight length, bob. The fast servo was bought for the walk's
   slew; the run is where it should pay. Expect the pantograph's work space
   (a 12 cm step at the crouch) to be the limit, and the `run_crouch` and
   `run_ext` genes to want the leg's reach.
6. **Gait changes.** `configs/skater_course3.json` profile (stand, walk, run,
   walk, stand) on the pantograph: the transition pitches are the score.
   If the run of step 5 does not exist, this is walk, stop, walk.
7. **Jump.** `configs/skater_hop.json` (hop and broad jump). The pantograph's
   two servos share every load, which is the argument for it; the hop is
   where that shows or does not. Expect the servo torque gene to want to
   move; let it, with its mass.
8. **Skate.** `configs/skater.json` (skate flat, cobblestone, slalom, stand)
   on the pantograph. Untouched since the leg changed. The crouch and the
   tread's position under the body at the skate stance need a snapshot
   (`tools/skater_snapshot.py`) before the round; the wide hip, if phase 1
   picked one, is the skating stance the hip was widened for.
9. **The traveller.** One course profile with every mode: stand, walk, run,
   walk, stop, skate, stop, hop. Scored as the course is, with the skate
   segment on the tread speed. Body locked; a long gains round. The clip is
   the deliverable.

## Order and cost

Each 40-generation round is 15-20 minutes on 12 workers; a seed search is
3-5 minutes. Phase 1 is about 2 hours of compute including the long round,
phase 2 about 3 hours. Phase 0's roll servo spec and fall direction come
before round A; the speed-under-load model and the rolled-plane IK check
before round 9 is believed.

## What is deliberately not in the plan

- GPU physics or a different simulator (memory: no GPU physics).
- A new leg topology. The pantograph is the leg until phase 2 says otherwise.
- Hip yaw. Turning stays tread differential plus fan yaw.
- Freeing the fan hardware spec: a catalogue choice, locked by group.
