# Skater skeleton: what was built from `skater_design.md`, and where it bends

Status: stage-1 skeleton, built for review before any evolution run. The
biped is untouched; the skater lives in `evol_mech/skater/` and is selected by
`"body": "skater"` in a run config (`configs/skater.json`).

Render the default body and an 8 s skate clip with

    .venv\Scripts\python tools\skater_snapshot.py            # -> runs/skater_skeleton/
    .venv\Scripts\python tools\view.py --body skater          # interactive, --objective crab|slalom|stand

## Design doc -> code

| Design doc | Where | Notes |
|---|---|---|
| Mass stays at the hip; every leg-joint motor in the hip cluster | `skater/structure.py` | Ten QDD motors (`leg_motor_torque` gene -> mass) are added to the hip body. Thigh and shin carry tube + pushrod + belt mass per metre only. |
| Each leg at most ~10-12 % of total mass, penalised in decode | `structure.leg_mass_fraction`, `rollout.LEG_MASS_PENALTY` | Default body: 7.7 % per leg. Excess above 12 % costs 5 fitness per unit fraction per leg. |
| No passive self-centering, alignment held actively | `skater/mjcf.py` | No caster offset anywhere: hip yaw axis passes through the ankle/sole centre. |
| Feet parallel unless steering, one turning centre (Ackermann) | `controller.act`, steering block | Each foot's ground velocity = v + omega x r_foot; hip yaw points the tread along it, tread speed is its magnitude. Straight ahead both yaws are exactly 0. Crab = both feet to the same yaw. |
| Hip yaw / roll / pitch, knee, ankle pitch; no ankle roll or yaw | `mjcf._leg` | Five hinges per leg, ranges +-90, +-35, -120..+60, 0..150, +-70 deg. Yaw axis vertical through the hip point. |
| Hip-mounted motors, knee by belt, ankle by pushrod | `mjcf._leg_actuators` | Simulated as position actuators on the joints with the QDD torque limit; the pushrod is represented by `ankle_lever` (torque x, speed /). The transmission's own mass is in the per-metre link mass. |
| QDD BLDC leg motors, steppers retired | `body/hardware.leg_motor_from_params` | 0.5-6 Nm peak, 66-300 g each. |
| Tread: rubber band under a flat sole, pulleys and idlers | `mjcf._foot`, `structure.foot_layout` | Row of 2-6 hinged rollers under a sole plate, radius = `pulley_radius`, spaced so they never overlap; rollers tied by joint equalities, one velocity actuator on the first roller. |
| Knee-mounted motor -> worm -> shin belt -> ankle-coaxial pulley | `body/hardware.tread_drive_from_params`, tread-drive geom at the knee | Speed = motor rpm / ratio x radius (default 2.4 m/s); pulley torque = P / omega_motor x ratio x 0.45; drive mass sits on the shin at the knee. Belt length constancy is implicit (no belt is simulated). |
| Worm does not back-drive: stopped tread = locked sole | `controller` tread tracking | Position-tracked velocity command (rate + angle-error term). Zero command holds the roller angle. Verified: parked body moves < 1 cm in 1 s. |
| No coasting: tread always driven at ground speed | `controller` tread block | Tread command = this foot's measured ground speed along its tread + a lead proportional to the force asked of it. It never steps to the target speed. |
| Shock path: belt is the fuse | not modelled | Torque limit on the roller actuator stands in for it. |
| Wheels as a `foot_type` alternative | `foot_type` = 1 | Heel and toe wheels, 1.5 x roller radius, same drive. |
| Two vectored fans, second (roll) vectoring axis | `mjcf._fan` | Pitch hinge then roll hinge, each with the 10 g servo model; fan mass includes both servos. The "two-axis gimbal vs fixed cant" question is left open: this is the gimbal. |
| Body planner | `controller.plan` | Task gives the objective and speed; slalom follows y = A sin(2 pi x / lambda) with a lateral-error correction. |
| Force allocation: treads first within mu N, fans the rest, all when airborne | `controller.act` allocation | Per-foot cap = min(`friction_margin` x mu N, tip-over limit); fans get the remainder of x and all of y. Airborne (hip above nominal + 3 cm) sets N = 0. |
| Suspension and leveling PD | `controller.act` leveling | Knee PD on height, hip roll PD on roll, hip pitch PD on pitch, ankle PD on pitch (ankle strategy), lean into acceleration. |
| Fan attitude: existing PD loops | same gains as the biped | Allocation tilt added on top. |
| Genome list | `skater/spec.py` | 42 genes in groups geometry, leg, foot, drivetrain, fan_spec, suspension, alloc, steer, fan_ctrl. `--lock-groups` works per group. |
| Derived masses with new entries | `skater/structure.py` | Hip motor cluster, tread motor + worm, shin belt, rollers, sole plate, tread rubber, both fan servos. Core, battery and fan placement reuse the biped rules. |
| MuJoCo tread as rollers with tied joints, one velocity actuator | as above | Plus `cone="elliptic"`, `impratio="10"`, `noslip_iterations="3"`: with the defaults the treads crept at low force. |
| Tasks skate_flat, skate_rough, slalom, crab, stand_perturbed | `skater/rollout.py`, `configs/skater.json` | Fitness: mean speed x upright fraction, ride-quality cost on RMS hip acceleration, lateral error and foot scrub on the slalom, drift on stand, -0.5 per fall, fan and motor energy, leg-mass penalty. Only feet (sole, rollers) may touch the floor. |

## Where the implementation departs from the doc, deliberately

1. **Ankle does not hold the sole level in world frame while planted.** The doc
   says the ankle "keeps the sole flat on slopes". A world-level feed-forward on
   a planted sole pushes the body the wrong way (toe-down when the body is
   nose-up lifts the front rollers); it made the first build fall in 0.3 s. The
   ankle is now square to the leg plus a balance PD on body pitch, and complies
   with slopes through that PD. Level-in-air feed-forward can come back when
   stage 2 adds swing legs.
2. **Tread force is capped by a tip-over limit, not only by friction.** Pushing
   at ground level with the centre of mass 40 cm up tips a 14 cm sole at about
   1.2 m/s^2. Beyond `TIP_MARGIN` x that, the remainder goes to the fans. This is
   the doc's "fans and treads push together" made explicit.
3. **Lean is a foot shift.** `lean_gain` swings the feet back under the lean
   (positive hip pitch per m/s^2 demanded) rather than only setting a body-pitch
   reference, so the inertial force lands mid-sole.
4. **The pushrod and belt are not simulated as mechanisms.** Joint actuators
   carry their torque limits; `ankle_lever` is the only pushrod geometry gene so
   far. Pushrod attachment offsets as listed in the doc's genome can be added
   once there is a reason to model their nonlinearity.
5. **No lateral tread compliance.** Rollers are rigid cylinders, so a yawed tread
   scrubs with full friction, which the slalom's scrub cost sees.

## Numbers for the default body

| | |
|---|---|
| Total mass | 2.74 kg |
| Hip incl. ten 2 Nm QDD motors | 1.52 kg |
| Per leg (thigh, shin + tread drive, foot) | 0.21 kg, 7.7 % |
| Fan, each, incl. two servos | 47 g, 5.2 N max |
| Tread | 2.4 m/s max, 11 N max, 0.20 Nm at the pulley, 78 g drive |
| Hip height in the parked crouch | 0.33 m (thigh 150, shin 150, knee 40 deg) |

Default-gain behaviour (untuned, seed 0, flat): hold 6 s with 3 cm drift,
stand with shoves 6 s, skate at 0.6 m/s for 8 s covering 4.3 m, skate at
1.2 m/s works with a lighter lean gain, crab at 0.3 m/s works for some gain
sets and tips for others.

## Open questions for review

- **Crab balance.** With both feet yawed 90 deg the fore/aft base is the 30 mm
  roller width and the ankle axis is sideways, so body pitch has no leg
  authority at all. Either the fans own pitch in crab, or crab uses less than
  90 deg of yaw, or the hips spread. The doc's "up to +-90 deg" needs one of
  these picked.
- **Fan vectoring.** Built as a two-axis gimbal (pitch then roll). Fixed cant
  with differential throttle would be lighter and is a small change here.
- **QDD mass.** Ten motors at 2 Nm are 1.3 kg of a 2.7 kg robot. The genome can
  shrink them (0.5 Nm -> 66 g each) but the hip will stay the heaviest part by
  design; whether hip yaw and roll really need the same motor as hip pitch and
  knee is a doc question.
- **Tread drive numbers.** Worm efficiency 0.45 and a 19 000 rpm motor cap are
  placeholders in `body/hardware.py`; replace with the chosen parts.
- **Pushrod offsets** as genes: not yet, see departure 4.
- **Ride-quality cost** is RMS hip acceleration including the spawn drop; fine
  for ranking, crude for absolute numbers.
