# Teaching a Custom Hexapod to Walk with Reinforcement Learning

**From a Fusion 360 CAD model to a walking 18-DOF hexapod in NVIDIA Isaac Sim — and every bug in between.**

This document chronicles the full debugging journey of my CS372 final project: training a
custom-designed 6-legged robot to follow velocity commands using PPO (rsl_rl) in Isaac Sim 5.1 /
Isaac Lab. The robot was designed in Fusion 360, exported to URDF with the `fusion2urdf` plugin,
and trained with a Direct-workflow RL environment adapted from a quadruped template.

What makes this project interesting isn't the happy path — it's that nearly every layer of the
pipeline (CAD export, physics asset, coordinate conventions, sensor wiring, reward design) hid a
bug, and each one produced misleading symptoms that pointed at the wrong layer. Below is each
problem in the order we hit it, how it was diagnosed, and how it was fixed.

**Stack:** Fusion 360 → fusion2urdf → URDF → Isaac Sim 5.1 (PhysX) → Isaac Lab DirectRLEnv →
rsl_rl PPO · Python/PyTorch · RTX 5060 Laptop (8 GB), 2048 parallel environments.

**Final result:** mean reward −18 → **+59.7**, episode length **999/1000** (never falls),
**~99% linear-velocity tracking** and ~98% yaw-rate tracking, a natural walking gait on all six
feet, trained in ~24 minutes (73.7M sim steps, ~55k steps/s).

---

## Problem 1 — The robot was 8× too heavy (steel by default)

**Symptom:** On spawn, joints ripped apart and the physics exploded into NaN.

**Cause:** `fusion2urdf` derives link masses from the *physical material assigned in Fusion 360*.
Nothing had been assigned, so every part exported as **steel** — a 5.7 kg base plate on hobby-servo
joint drives.

**Fix:** Assign ABS plastic to every component in Fusion, re-export. Base link went 5.72 kg → 0.77 kg,
total robot ~2.97 kg. As a bonus, the export was later verified against first principles: computing
plate/rod inertia estimates (`I = m/12·(a²+b²)`) from the joint-origin geometry matched the exported
inertia tensors within ~10%, confirming both masses and inertias.

**Lesson:** CAD exporters inherit whatever the CAD file says. Sanity-check exported mass properties
against a back-of-envelope calculation before blaming your simulator.

## Problem 2 — Inertia units (kg·mm² vs kg·m²)

**Symptom:** Inertias 10⁶× too large in early exports.

**Cause:** `fusion2urdf` exports inertia in kg·mm²; URDF expects kg·m².

**Fix:** A small script multiplying all `<inertia>` values by 1e-6 — but critically, we later
verified a *new* export already had correct SI values, so running the fix script again would have
corrupted it. The verification (radius-of-gyration checks against part dimensions) became a
standard step after every export.

**Lesson:** Never blindly re-apply a data-migration fix. Verify units from the numbers themselves —
a 0.1 kg part with a 3 cm radius of gyration must have I ≈ 1e-4 kg·m², full stop.

## Problem 3 — All 18 joints were `continuous` with no limits

**Symptom:** The RL policy could fold legs through the body (self-collision was disabled), and
`soft_joint_pos_limit_factor` did nothing.

**Cause:** `fusion2urdf` exports joints as `type="continuous"` (unlimited) when no limits are set
in Fusion.

**Fix:** Scripted conversion to `type="revolute"` with per-group limits and effort/velocity caps
matched to the Isaac actuator config. Limits were later refined to the real servo hardware's
travel (coxa ±45–90° asymmetric per mount, femur −55..−20°, tibia 50..120°).

## Problem 4 — The big one: every rotated joint had a wrong pivot and axis

This was the root cause of *months* of "the robot's joints look disconnected" mystery across
earlier project iterations.

**Symptom:** The robot looked perfect at spawn but collapsed the moment drives engaged. In the
Physics Inspector, tibias rotated around a point *in mid-air ~7 cm above the knee*, and corner-leg
joints swung on strange diagonal arcs.

**Diagnosis (the fun part):** Because `fusion2urdf` writes meshes in global CAD coordinates, the
URDF is internally self-consistent even when it's wrong — the visual origins always equal minus the
cumulative joint origins, so nothing looks broken until physics moves a joint. We diagnosed it by
**measuring the actual mesh geometry against the joint definitions**, with three independent
estimators:

1. **Contact-ring centroids** — where femur and tibia meshes nearly touch (KD-tree proximity
   queries) is by construction centered on the real hinge.
2. **Concentric-circle fitting** — servo hinges leave circular features (shaft holes, horn discs);
   fitting circle centers via support functions and radial-concentration search located hinge axes.
3. **Part-identity registration** — all six femurs are copies of one part. Rigidly registering
   (ICP with yaw/mirror seeds, 0.000 mm residual) leg 1's mesh onto each other leg and mapping
   leg 1's *known-good* hinge point through the transform gave each leg's true hinge — with
   built-in control groups (joints known to be correct had to come out at 0 error, and did).

**Findings:** every knee pivot was displaced **70–86 mm** (outside both parts!), two pairs of hip
pivots were off by 3.5 mm and 43 mm, and all eight corner-leg axes were **pre-rotated by exactly
45°** — the leg-splay angle. Classic `fusion2urdf` occurrence-transform bug: components that are
*rotated in the CAD assembly* get their joint origins/axes exported without the occurrence
transform applied. The middle legs (unrotated occurrences) were the only correct ones.

**Fix:** Scripted surgery on the URDF — moved all 12 leg-joint origins to the measured true hinge
points, de-rotated the 8 corner axes, and compensated every affected link's visual/collision/
inertial origins so the meshes stayed within **0.2 µm** of their original positions. Verified by
re-running the independent contact measurement: all 18 anchors within 0.6 mm of the mesh hinges.

**Lesson:** "Internally consistent" ≠ "correct". When a kinematic model misbehaves, measure the
geometry itself — the meshes carry the ground truth about where the hinges physically are. And
always build control groups into your estimators (known-good joints must measure as zero error)
before trusting them on the unknowns.

## Problem 5 — Y-up robot, Z-up code

**Symptom:** Robot spawned lying on its side in training — even after baking an upright rotation
into the USD. Meanwhile the "is the robot flat?" reward read ~0 (perfectly level!), masking the
problem.

**Cause:** Two conventions colliding. (1) Isaac Lab *overwrites* the USD root pose with the
config's `init_state.rot` on every reset — orientation baked into the USD is ignored. (2) The
fusion2urdf robot is authored **Y-up** (all hips at constant Y; base-plate normal = body +Y), but
the reward/termination code inherited from a Z-up quadruped template used index 2 (body Z) as "up"
everywhere. With the identity spawn, body-Z happened to align with gravity, so the flatness check
was satisfied by a sideways robot.

**Fix:** Spawn quaternion `(0.7071068, 0.7071068, 0, 0)` (+90° about X, mapping body +Y → world +Z)
in the articulation config, plus remapping every up-axis reference in the env: tilt termination and
vertical-velocity penalty to index 1, ground-plane velocity/flatness terms to indices `[0, 2]`,
yaw rate to index 1. (World-frame quantities like base height stay on world Z.)

**Lesson:** A reward term reading "perfect" can be the bug. When a check passes suspiciously
easily, ask what frame it's measured in.

## Problem 6 — The termination that read the wrong body

**Symptom:** Robot spawned a few cm up (normal settling), landed on its feet… and instantly reset,
every episode.

**Cause:** An index-space bug. The contact sensor tracked only the six tibias, but the
"base hit the ground" termination indexed the sensor's force tensor with the **articulation's**
`base_link` index (0) — which in the *sensor's* index space was Tibia_1. The check actually meant
"terminate when foot 1 presses the ground harder than 1 N," i.e. always, on landing. It had been
masked earlier by the sideways spawn (that foot never touched cleanly).

**Fix:** Track all bodies in the contact sensor and resolve `base_link`'s index *from the sensor*,
so termination truly measures the base. Episode length immediately went ~40 → 739+.

**Lesson:** When two components each maintain their own body ordering, never let an index cross
between them. Resolve names in the index space you're about to use.

## Problem 7 — Reward hacking, round 1: the robot dragged itself on its thighs

**Symptom:** Training "succeeded" (97% velocity tracking!) but the gait was grotesque — the robot
splayed flat and propelled itself on its femurs, feet barely used.

**Cause:** The reward only asked for velocity tracking + staying upright. Two design bugs made
dragging optimal: the feet-air-time reward required a foot to be airborne **>0.5 s** to earn
anything (a quadruped constant — hexapod tripod-gait swings last ~0.2–0.3 s), so it was
*structurally negative* and actively discouraged stepping; and non-foot links touching the ground
cost nothing.

**Fix:** Lowered the air-time threshold to 0.3 s and doubled its weight, and added an
**undesired-contact penalty** (−0.5 per base/coxa/femur body pressing the ground). Retrained
from scratch — the gait switched to walking on tibia tips.

## Problem 8 — Reward hacking, round 2: the five-legged gait

**Symptom:** With the improved reward, every play-test robot held **one middle femur pinned at its
raised limit**, that foot never touching the ground — walking on 5 legs with one leg as a balance
arm. Identical on every robot (they share one deterministic policy).

**Diagnosis:** First ruled out an asset asymmetry analytically — rotating each femur's knee offset
about its actual URDF axis gave *bit-identical* knee heights for all six legs at the default and
both limit extremes. So it was learned. The loophole: the air-time reward only pays **at
touchdown**. A foot that never lands generates no contact events and contributes exactly 0 —
parking a leg was reward-neutral, and coordinating 5 legs is easier than 6.

**Fix:** A **dangling-foot penalty**: any foot whose `current_air_time` exceeds 1.0 s accrues
−0.25/s. Real swing phases (~0.2–0.3 s) never touch the threshold, so it surgically punishes only
parked legs.

## Problem 9 — Reward hacking, round 3: the grounded shuffle

**Symptom:** The dangling-foot penalty worked — no more parked leg — but the gait got *worse*.
The robot now vibrated its legs at high frequency, taking tiny partial steps, inching forward with
no visible swing at all, its body hugging the ground.

**Diagnosis:** TensorBoard told the story that three runs of tuning had missed: `feet_air_time`
had been **negative for its entire history, in every run**, never once crossing zero. The reward
paid for a foot only if it stayed airborne past a threshold — set to 0.5 s, then 0.3 s — but this
robot's legs are short and its real swings last only ~0.1–0.2 s. So no achievable swing ever
earned the reward. Worse, once the dangling penalty was added, the two terms *bracketed* stepping
out of existence: air time under 0.3 s cost reward, air time over 1.0 s cost reward, and nothing
in between earned any. The globally optimal policy under that landscape is exactly what emerged —
feet glued down, shuffling as fast and shallow as possible, crouched low for stability. Round 2's
fix had removed one bad local optimum and revealed a bad *global* one hiding underneath.

**Fix:** Three coordinated changes so that, for the first time, both shaping terms pointed *toward*
stepping: lowered the air-time threshold to **0.1 s** (a real 0.1–0.2 s swing now earns reward, and
only a truly parked >1.0 s leg gets penalized), raised the action-rate penalty 2.5× to tax the
jitter directly, and extended the default knee angle (115° → 105°) so the robot's resting stance
stands taller. The retrain produced a visibly natural walk. `feet_air_time` went **positive for the
first time** (−0.019 → +0.002), the crouch penalty halved, and — the satisfying part — the
action-rate penalty *improved* despite being weighted 2.5× heavier, meaning the motion was
genuinely smoother rather than merely re-scaled.

**Lesson (all three rounds):** RL optimizes exactly what you wrote, not what you meant. Every "the
robot found a weird trick" moment traced to a reward term that was silently zero or negative along
the behavior I actually wanted — and fixing one exposed the next. The single most diagnostic habit
was reading the *per-term* episode logs: the sign of `feet_air_time`, sitting quietly negative for
four runs, was the whole story the entire time.

---

## Results

| Metric | Start | Final |
|---|---|---|
| Mean reward | −18.6 | **+59.7** |
| Mean episode length | ~40 steps | **999 / 1000 cap** (never falls) |
| Linear-velocity tracking | ~0 | **1.98 / 2.0 (~99%)** |
| Yaw-rate tracking | ~0 | **0.98 / 1.0 (~98%)** |
| Policy action std | 1.0 | 0.06 |
| Training time | — | ~24 min (2048 envs, 8 GB laptop GPU) |

Getting the numbers took one training run; getting a *natural gait* took four, each one fixing a
reward term that had quietly encoded the wrong incentive (see Problems 7–9). The final policy walks
on its feet with a real swing phase, all six legs participating.

## The meta-lessons

1. **Bugs stack.** The asset bugs (pivots, axes, mass) had to be fixed before the convention bugs
   (Y-up, sensor indices) were even *visible*, and those had to be fixed before the reward bugs
   mattered. Debugging order was forced from the physics up.
2. **Measure, don't stare.** The decisive diagnoses came from scripts that measured geometry
   (KD-trees, circle fits, ICP registration) and from per-term reward logs — not from looking
   harder at the viewport.
3. **Control groups for everything.** Every estimator we trusted was first validated on joints we
   knew were correct. The one estimator we skipped that step for produced a wrong "fix" that had
   to be reverted.
4. **Keep a lab notebook.** Every fix, dead end, and gotcha was logged in a running project-context
   file. On a multi-week project with breaks, that file *was* the difference between resuming in
   minutes vs. re-debugging old ground.
