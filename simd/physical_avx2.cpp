#include "df/simd_inverse.h"
namespace rrtpff {
using namespace wide;
namespace {
unsigned collisions(const batch_df::Physical &p, const SphereGeometry &g, std::size_t id,
                    unsigned active, std::size_t &tests) {
    const auto &n = g.nodes[id];
    F inside = F(0) <= F(0);
    for (int k = 0; k < g.dimension; ++k)
        inside = inside & (F(n.lower[k]) <= p.x[k]) & (p.x[k] <= F(n.upper[k]));
    active &= bits(inside);
    if (!active)
        return 0;
    if (n.left) {
        unsigned hit = collisions(p, g, n.left, active, tests);
        return hit | collisions(p, g, n.right, active & ~hit, tests);
    }
    unsigned hit = 0;
    for (std::size_t i = n.begin; i < n.end && (active & ~hit); ++i) {
        const auto &s = g.spheres[i];
        F distance;
        for (int k = 0; k < g.dimension; ++k) {
            F d = p.x[k] - F(s.center[k]);
            distance = distance + d * d;
        }
        F r = F(s.radius) + F(g.robot_radius);
        hit |= bits(distance <= r * r) & active;
        ++tests;
    }
    return hit;
}
void check(batch_df::Physical &p, const PhysicalBatchConfig &c, unsigned active,
           const SphereGeometry &g, PhysicalBatch &out, bool state_only) {
    out.df_mask = state_only ? active : bits(p.valid) & active;
    F state_ok = F(0) <= F(0), control_ok = state_ok;
    for (int k = 0; k < c.state_dim; ++k) {
        state_ok = state_ok & finite(p.x[k]) & (F(c.state_lower[k]) <= p.x[k]) &
                   (p.x[k] <= F(c.state_upper[k]));
        p.x[k].store(out.states[k]);
    }
    for (int k = 0; k < c.control_dim; ++k) {
        control_ok = control_ok & finite(p.u[k]) & (F(c.control_lower[k]) <= p.u[k]) &
                     (p.u[k] <= F(c.control_upper[k]));
        p.u[k].store(out.controls[k]);
    }
    out.state_mask = bits(state_ok) & active;
    out.control_mask = state_only ? active : bits(control_ok) & active;
    unsigned eligible = out.df_mask & out.state_mask & out.control_mask;
    out.sphere_tests = 0;
    out.collision_mask = 0;
    if (eligible && !g.nodes.empty())
        out.collision_mask = collisions(p, g, 0, eligible, out.sphere_tests);
    out.valid_mask = eligible & ~out.collision_mask;
}
void inverse(const batch_df::Jet &j, const PhysicalBatchConfig &c, int branch, unsigned active,
             const SphereGeometry &g, PhysicalBatch &out) {
    batch_df::Physical p;
    switch (c.robot) {
    case BatchRobot::Unicycle:
        p = batch_df::unicycle(j, c, branch);
        break;
    case BatchRobot::Quad2D:
        p = batch_df::quad2d(j, c);
        break;
    case BatchRobot::Quad3D:
        p = batch_df::quad3d(j, c);
        break;
    }
    check(p, c, active, g, out, false);
}
} // namespace
void physicalBatchAVX2(const JetBatch &jet, const PhysicalBatchConfig &c, int branch,
                       unsigned active, const SphereGeometry &g, PhysicalBatch &out) {
    batch_df::Jet j;
    for (int d = 0; d < 5; ++d)
        for (int k = 0; k < 4; ++k)
            j.d[d][k] = F::load(jet.values[d][k]);
    inverse(j, c, branch, active, g, out);
}
void curveBatchAVX2(const BernsteinCurve &curve, const float *times, const PhysicalBatchConfig &c,
                    int branch, unsigned active, const SphereGeometry &g, PhysicalBatch &out) {
    batch_df::Jet j;
    F t = F::load(times), s = F(1) - t;
    for (int d = 0; d < 5; ++d)
        for (int k = 0; k < 4; ++k)
            j.d[d][k] = F(0);
    for (int d = 0; d <= curve.derivative_order; ++d)
        for (int k = 0; k < curve.dim; ++k) {
            F work[8];
            const int degree = 2 * curve.order - 1 - d;
            if (degree < 0) continue;
            for (int i = 0; i <= degree; ++i)
                work[i] = F(curve.coefficients[d][k][i]);
            for (int n = degree; n > 0; --n)
                for (int i = 0; i < n; ++i)
                    work[i] = s * work[i] + t * work[i + 1];
            j.d[d][k] = work[0];
        }
    inverse(j, c, branch, active, g, out);
}
void stateBatchAVX2(const PhysicalBatchConfig &c, unsigned active, const SphereGeometry &g,
                    PhysicalBatch &out) {
    batch_df::Physical p;
    for (int k = 0; k < c.state_dim; ++k)
        p.x[k] = F::load(out.states[k]);
    check(p, c, active, g, out, true);
}
} // namespace rrtpff
