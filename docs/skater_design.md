# Skater mech: stage-1 design

Status: draft, replaces the crank-driven 4-bar biped as the target body.
The evolution stack (GA, worker pool, rollout, data store, hardware models)
carries over; the body, genome, controller and tasks are new.

## Goal

A fan-assisted biped that travels mostly by **powered feet with the legs held
still**, like a VOTOMS ground-roller dash, and can also walk, run and jump,
including sideways. The fans and the feet push **together**: neither leads,
a single controller splits the demanded body force between them.

Stage plan:

1. **Skate**: tread feet, legs as active suspension and leveling, steering by
   hip yaw, fan/tread force sharing. Evolve morphology and gains for fast,
   smooth travel over rough ground.
2. **Walk**: stepping and side steps on the same hardware.
3. **Run and jump**: mostly controller work (likely RL for the policy, with
   evolution kept for morphology and gains).

## Design rules

- **Mass stays at the hip.** Every leg-joint motor sits in the hip cluster;
  below it are links, rods and belts. Target: each leg (thigh, shin, foot,
  tread) at most ~10-12 % of total mass. Distal mass is penalised in decode,
  not just by physics.
- **No passive self-centering geometry.** The robot moves in every direction,
  so alignment is held actively, not by caster trail.
- **Feet stay parallel unless steering.** Both feet are commanded from one
  turning centre (Ackermann); going straight they are exactly parallel, so
  the legs never fight each other through the hip.

## Leg (per side, both identical)

| Joint | Actuation | Notes |
|---|---|---|
| Hip yaw | motor in hip | also the steering joint for skating; the leg below does not twist, so foot yaw = hip yaw |
| Hip roll | motor in hip | lateral leveling, side steps |
| Hip pitch | motor in hip | |
| Knee | hip-mounted motor, belt or pushrod | |
| Ankle pitch | hip-mounted motor, long pushrod | push-off for running/jumping; keeps the sole flat on slopes |
| Ankle roll | none (optional passive elastomer) | hip roll covers leveling |
| Ankle yaw | none | steering is at the hip |

Leg motors: quasi-direct-drive BLDC (low ratio planetary) so the legs can run,
jump and absorb landings. Steppers are retired from the leg joints.

## Foot: tread

A short rubber tread under a flat sole, front and rear pulleys plus idlers.

**Drive train.** Small fast BLDC mounted at the knee (on the shin) ->
self-locking worm reduction -> belt down the shin -> pulley **coaxial with
the ankle pitch axis** -> tread drive pulley. Coaxial routing keeps belt
length constant through ankle motion; the residual coupling (ankle rotation
creeps the tread by `r_pulley * d_ankle`) is cancelled in the controller.

**Why a worm.** It does not back-drive, so a stopped tread is a locked rubber
sole: full grip for standing, walking and landing with no holding current.
The high ratio lets the motor be small in torque. Accepted cost: efficiency
roughly 40-50 % (self-locking needs a low lead angle); leg motors and fans
dominate the power budget anyway.

Consequences to design for:

- **Speed comes from motor rpm.** Tread speed `v = rpm_motor / ratio * r`.
  Example: r = 20 mm, v = 2 m/s needs 955 rpm at the pulley; at 15:1 that is
  ~14 000 rpm at the motor. So ratio stays modest (roughly 10-20:1) and the
  motor is a small high-kv BLDC. Ratio and pulley radius are genes.
- **Motor power is not smaller.** Torque multiplies, power does not: electrical
  power is mechanical power / efficiency, so the motor's power rating grows
  about 2x versus a direct belt.
- **No coasting.** The tread cannot freewheel, so it must always be driven at
  ground speed: when the fans push, the treads are commanded to match, and
  downhill the treads hold back. This is what makes the fan/tread motion
  coordinated rather than one dragging the other.
- **Shock path.** The worm cannot slip, so impact and stall loads go into its
  teeth. The shin belt is the compliant element and fuse: tension and tooth
  profile set the slip torque.

Tread vs wheels: the tread wins on rough and soft ground and as a stable flat
sole; it costs more rolling resistance and scrubs when the foot yaws. Wheels
(heel and toe, hub motors) stay available as a `foot_type` gene so evolution
can compare them on mixed terrain.

## Fans

Two vectored fans on the core, as now. Sideways boosts need lateral thrust,
so each fan gets a second vectoring axis (roll) in addition to pitch.
Open question: two-axis gimbals versus fixed cant angles with differential
throttle.

## Controller (stage 1)

1. **Body planner**: desired body velocity (x, y, yaw) and height.
2. **Force allocation**: demanded horizontal force is split between treads and
   fans. Treads first, limited by `mu * N` on each foot; fans take the rest,
   all of it when airborne, and the sideways component during boosts. The
   split weights and the friction margin are genes.
3. **Steering**: one turning centre -> both hip yaw angles (Ackermann) and
   each tread's speed (inner foot slower). Crab motion: both feet to the same
   yaw, up to +-90 deg.
4. **Suspension and leveling**: hip roll, hip pitch, knee and ankle hold body
   height, roll and pitch with PD (later impedance) control; leaning into
   acceleration like a skater is part of the pitch reference.
5. **Fan attitude**: the existing PD loops, now sharing pitch authority with
   the legs.

## Genome (stage 1)

- Geometry: hip width, thigh and shin length, crouch (nominal knee angle),
  pushrod attachment offsets, core size and battery, fan placement.
- Foot: type (tread or wheels), sole length, pulley radius, worm ratio.
- Suspension and leveling gains.
- Allocation weights, friction margin, steering gains.
- Fan spec and attitude gains (as now).

Masses stay derived (`body/structure.py`), with new entries for the hip motor
cluster, pushrods, shin belt, knee tread motor and worm, and the tread.

## Simulation notes

- MuJoCo has no native tracks. Model the tread as a row of small rollers
  under the sole, joints tied by a fixed tendon / equality constraint and
  driven through one velocity actuator with a torque limit; a non-back-
  drivable joint is a stiff position hold when the command is zero.
- Elliptic friction cones for the roller contacts.

## Fitness tasks (stage 1)

- `skate_flat`: forward speed, penalise body acceleration (ride quality).
- `skate_rough`: same on cobblestone / random heightfield.
- `slalom`: follow a curved path, penalise lateral error and foot scrub.
- `crab`: sideways travel.
- `stand_perturbed`: kept from the current set.
- Energy cost on fans, tread motors and leg motors.

## Retired

Crank-rocker 4-bar legs, stepper leg drive, the open-loop crank gait.
