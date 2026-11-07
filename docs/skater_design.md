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
| Hip yaw | ~~motor in hip~~ none (stage 1 decision) | was the steering joint for skating; traded for the roll servo so the hip keeps three servos. Turning is tread differential + fan yaw; see `skater_review.md` |
| Hip roll | motor in hip | lateral leveling, side steps, foot under the body for walking and landing |
| Hip pitch | motor in hip | |
| Knee | hip-mounted motor, belt or pushrod | |
| Ankle pitch | hip-mounted motor, long pushrod | push-off for running/jumping; keeps the sole flat on slopes |
| Ankle roll | passive spring (built) | lets the sole lie flat under a rolled leg |
| Ankle yaw | none | steering is at the hip |

Leg motors: quasi-direct-drive BLDC (low ratio planetary) so the legs can run,
jump and absorb landings. Steppers are retired from the leg joints.

### Leg drives (built)

The built leg uses hobby servos (cheap, procurable) and has two drives,
selected by the `leg_drive` gene:

- **0, four-bar (the original):** the thigh servo drives the thigh directly;
  the shin servo, in the hip too, drives the shin through a horn, a coupler
  down the thigh and a pin tab on the shin (LEO's parallelogram when coaxial,
  staggered by default). The knee is a hinge the controller sets to what the
  four-bar would give.
- **1, pantograph (the 2026-11-07 sketch, `docs/leg_layout_sketch.png`):**
  both leg servos sit under the
  hip bracket at the same height, `servo_dx` apart. Each horn (`horn_len`)
  drives a rod (`rod_len`) to a common pin, the knee node, where the rods
  cross and are pinned; each rod continues `rod_ext` past the pin (the front
  servo's rod to the rear lower node, the rear servo's to the front) and two
  links of that length close a parallelogram at the ankle node, which
  therefore sits at the vector sum of the two rod extensions. A strut
  (`strut_len`) continues the rear rod's direction to the same passive sprung
  foot. The horn angles place the pin (a five-bar); the pantograph carries
  that to the foot; both servos share every load. It is built in MuJoCo as
  the real closed chain (two `connect` equalities per leg), with the servo
  actuators on the horn hinges, so the load split and the fold stops are the
  linkage's own. Reach is the straight-down drop the horns can set with
  every link clear of its fold margin; the parked crouch stands at reach x
  cos(crouch / 2), as the two-link leg does. Its two roll pivots sit at the
  hip centre (ROLL_PIVOT_Y = 12 mm apart, the sketch's two bolts) and each
  bracket runs out to its leg plane, carrying that leg's two servos; the
  bracket is the roll lever, so the roll ratio is derived, (hip_width / 2 -
  ROLL_PIVOT_Y) / horn_len, and the `roll_ratio` gene is ignored on this
  body (on a 10 cm hip with 6 cm horns it is 0.64, the roll servo geared
  down; a wider hip or a shorter horn gears it up, and the horn is shared
  with the leg). The roll servos sit at the centre, one behind the other on
  their common forward axis, horn horizontal and a vertical link down to the
  bracket's outboard end, drawn as a light passive loop; the ratio is
  reflected onto the roll hinge (torque x ratio, stiffness x ratio^2, speed
  and range / ratio) rather than carried by that loop. Because the pivot is
  inboard, rolling the leg swings its hip point up or down by arm x
  sin(roll), and the controller's roll solve accounts for it. On the four-bar
  body the pivot stays at the leg plane and `roll_ratio` stands in for an arm.

All three horns per leg are one part: `horn_len` long and flexing in their
sweep plane with stiffness `10^horn_k_log10` N m/rad about the servo axis
(stiff out of plane, so the kinematics stay planar). That flexure is the
series elasticity: the servo's position stiffness (stall torque over a 4 deg
deadband) and the horn's spring act in series, and the actuator's stiffness
is their series sum. One stiffness for all three keeps the bill of materials
to one horn; independent values are a later experiment. The horn is capped
at 25 mm (gene range 10-25 mm, since 2026-11-07): a short horn is a real
hobby-servo horn, bends further for the same stiffness, and leaves the lever
ratio to the rods. With it the pantograph reaches a 12 cm step only as short
rods (8-10 cm) with long extensions (16-20 cm) that gear the horn up, and the
leg comes out long (40-50 cm reach); the foot-per-horn gearing lands near the
6 cm horn's (~0.06 m/rad), so the foot's stiffness and the servo's load are
about the same and it is the body's geometry that changes. The earlier
pantograph archetypes (6 cm horns, `skater_walker_panto*.json` up to 4) are
history under the cap: `encode` clamps their horn and the leg no longer
reaches a step, so they do not build.

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
- Leg drive: `leg_drive` (four-bar or pantograph), and for the pantograph
  `servo_dx`, `horn_len`, `rod_len`, `rod_ext`, `strut_len`; the hip roll
  lever `roll_ratio`; the horns' flexural stiffness `horn_k_log10`.
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
