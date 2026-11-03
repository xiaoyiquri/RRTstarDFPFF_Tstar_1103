# Differential-flatness layer

The DF layer is the boundary between a flat-space PFF trajectory and a
DynoBench-compatible physical state/control. It does not implement RRT*, PFF,
SIMD, or collision geometry.

`FlatState<Dim, Order>` contains derivatives `0 ... Order - 1` and
`FlatControl<Dim>` contains derivative `Order`. A trajectory is evaluated at
one time with `evaluateState(t)` and `evaluateControl(t)` before inverse DF is
called.

The physical-to-flat direction requires explicit lifting data:

* `UnicycleLift::signed_velocity` selects the edge's fixed forward/reverse
  branch.
* `Quad2DLift` supplies acceleration and jerk.
* `Quad3DLift` supplies acceleration, jerk, yaw rate, yaw acceleration, and
  yaw jerk for the complete `FlatState<4, 4>`.

Missing lifting data is rejected. No derivative is silently filled with zero.

Unicycle reverse edges carry `UnicycleTrajectoryMetadata`. The inverse uses
`atan2(dy, dx)` for the flat velocity direction, adds `pi` for the physical
heading on a reverse branch, and rejects speeds at or below
`velocity_epsilon`. A trajectory is also rejected if its initial speed does
not agree with `signed_start_velocity` or if its velocity polynomial reaches a
zero-speed stationary point.

Quad2D follows the DynoBench no-drag model. Quad3D uses the audited DynoBench
state layout `(p, qx, qy, qz, qw, v_world, omega_body)`, analytic derivatives
of the thrust frame, and the B0 motor allocation matrix; finite differences are
not used for angular velocity or angular acceleration.
