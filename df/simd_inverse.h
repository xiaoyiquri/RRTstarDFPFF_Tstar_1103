#pragma once
#include "simd/float8.h"
#include "simd/physical_batch.h"
namespace rrtpff::batch_df {
using namespace wide;
struct Jet {
    F d[5][4];
};
struct Physical {
    F x[13], u[4];
    F valid{1};
};
inline F norm(V3 a) { return sqrt(dot(a, a)); }
inline F norm2(F x, F y) { return sqrt(x * x + y * y); }
inline F allFinite(const Jet &j, int dim, int order) {
    F ok = F(0) <= F(0);
    for (int d = 0; d <= order; ++d)
        for (int k = 0; k < dim; ++k)
            ok = ok & finite(j.d[d][k]);
    return ok;
}
inline Physical unicycle(const Jet &j, const PhysicalBatchConfig &c, int branch) {
    Physical p;
    F vx = j.d[1][0], vy = j.d[1][1], q = vx * vx + vy * vy, speed = sqrt(q);
    p.valid = allFinite(j, 2, 2) & (speed > F(c.epsilon)) & (speed <= F(c.max_velocity));
    p.x[0] = j.d[0][0];
    p.x[1] = j.d[0][1];
    p.x[2] = wrap(atan2(vy, vx) + F(branch < 0 ? 3.141592653589793f : 0.f));
    p.u[0] = speed * F(static_cast<float>(branch));
    p.u[1] = (vx * j.d[2][1] - vy * j.d[2][0]) / q;
    p.valid = p.valid & (abs(p.u[1]) <= F(c.max_omega));
    return p;
}
inline Physical quad2d(const Jet &j, const PhysicalBatchConfig &c) {
    Physical p;
    F ax = j.d[2][0], az = j.d[2][1], fx = ax, fz = az + F(c.gravity);
    F jx = j.d[3][0], jz = j.d[3][1], sx = j.d[4][0], sz = j.d[4][1];
    F q = fx * fx + fz * fz, n = sqrt(q), numerator = fx * jz - fz * jx;
    F omega = numerator / q;
    F alpha = ((fx * sz - fz * sx) * q - numerator * F(2) * (fx * jx + fz * jz)) / (q * q);
    F thrust = F(c.mass) * n, torque = F(c.inertia2) * alpha;
    p.x[0] = j.d[0][0];
    p.x[1] = j.d[0][1];
    p.x[2] = atan2(-fx, fz);
    p.x[3] = j.d[1][0];
    p.x[4] = j.d[1][1];
    p.x[5] = omega;
    p.u[0] = (thrust + torque / F(c.arm)) / F(c.mass * c.gravity);
    p.u[1] = (thrust - torque / F(c.arm)) / F(c.mass * c.gravity);
    p.valid = allFinite(j, 2, 4) & (n > F(c.epsilon)) &
              (norm2(p.x[3], p.x[4]) <= F(c.max_velocity)) &
              (norm2(ax, az) <= F(c.max_acceleration)) & (norm2(jx, jz) <= F(c.max_jerk)) &
              (abs(omega) <= F(c.max_omega)) & (abs(alpha) <= F(c.max_alpha)) &
              (thrust <= F(c.max_thrust));
    return p;
}
struct Frame {
    V3 v, d, dd;
};
inline Frame normalize(Frame f, F epsilon, F &valid) {
    F n = norm(f.v);
    valid = valid & (n > epsilon);
    V3 b = f.v / n;
    F ndot = dot(b, f.d);
    V3 bd = (f.d - b * ndot) / n;
    V3 bdd = (f.dd - b * dot(b, f.dd)) / n - b * dot(bd, bd) - bd * (F(2) * ndot / n);
    return {b, bd, bdd};
}
inline Frame crossFrame(Frame a, Frame b) {
    return {cross(a.v, b.v), cross(a.d, b.v) + cross(a.v, b.d),
            cross(a.dd, b.v) + cross(a.d, b.d) * F(2) + cross(a.v, b.dd)};
}
inline V3 jet3(const Jet &j, int d) { return {j.d[d][0], j.d[d][1], j.d[d][2]}; }
inline Physical quad3d(const Jet &j, const PhysicalBatchConfig &c) {
    Physical p;
    p.valid = allFinite(j, 4, 4) & (abs(j.d[0][3]) <= F(1000));
    V3 a = jet3(j, 2), jerk = jet3(j, 3), snap = jet3(j, 4),
       force = a + V3{F(0), F(0), F(c.gravity)};
    Frame b3 = normalize({force, jerk, snap}, F(c.epsilon), p.valid);
    F s, co;
    sincos(j.d[0][3], s, co);
    F yd = j.d[1][3], ydd = j.d[2][3];
    Frame heading{{co, s, F(0)},
                  {-s * yd, co * yd, F(0)},
                  {-co * yd * yd - s * ydd, -s * yd * yd + co * ydd, F(0)}};
    Frame b2 = normalize(crossFrame(b3, heading), F(c.epsilon), p.valid);
    Frame b1 = crossFrame(b2, b3);
    Frame b[3] = {b1, b2, b3};
    V3 omega, alpha;
    for (int k = 0; k < 3; ++k) {
        int i = (k + 2) % 3, l = (k + 1) % 3;
        omega[k] = (dot(b[i].v, b[l].d) - dot(b[l].v, b[i].d)) * F(.5f);
        alpha[k] = (dot(b[i].v, b[l].dd) - dot(b[l].v, b[i].dd)) * F(.5f);
    }
    V3 jw{omega.x * F(c.inertia[0]), omega.y * F(c.inertia[1]), omega.z * F(c.inertia[2])};
    V3 torque =
        V3{alpha.x * F(c.inertia[0]), alpha.y * F(c.inertia[1]), alpha.z * F(c.inertia[2])} +
        cross(omega, jw);
    F thrust = F(c.mass) * norm(force), wrench[4] = {thrust, torque.x, torque.y, torque.z};
    for (int i = 0; i < 4; ++i)
        for (int k = 0; k < 4; ++k)
            p.u[i] = p.u[i] + F(c.wrench_inverse[i][k]) * wrench[k];
    for (int k = 0; k < 3; ++k) {
        p.x[k] = j.d[0][k];
        p.x[7 + k] = j.d[1][k];
        p.x[10 + k] = omega[k];
    }
    // Largest quaternion component avoids divisions by zero near 180 degrees.
    F r[3][3];
    for (int i = 0; i < 3; ++i)
        for (int k = 0; k < 3; ++k)
            r[i][k] = b[k].v[i];
    F diag[4] = {F(1) + r[0][0] - r[1][1] - r[2][2], F(1) - r[0][0] + r[1][1] - r[2][2],
                 F(1) - r[0][0] - r[1][1] + r[2][2], F(1) + r[0][0] + r[1][1] + r[2][2]};
    F best = diag[0], idx = F(0);
    for (int k = 1; k < 4; ++k) {
        F take = diag[k] > best;
        best = select(take, diag[k], best);
        idx = select(take, F(float(k)), idx);
    }
    F denom = F(2) * sqrt(max(best, F(0))), q[4];
    F cases[4][4] = {{denom * F(.25f), (r[0][1] + r[1][0]) / denom, (r[0][2] + r[2][0]) / denom,
                      (r[2][1] - r[1][2]) / denom},
                     {(r[0][1] + r[1][0]) / denom, denom * F(.25f), (r[1][2] + r[2][1]) / denom,
                      (r[0][2] - r[2][0]) / denom},
                     {(r[0][2] + r[2][0]) / denom, (r[1][2] + r[2][1]) / denom, denom * F(.25f),
                      (r[1][0] - r[0][1]) / denom},
                     {(r[2][1] - r[1][2]) / denom, (r[0][2] - r[2][0]) / denom,
                      (r[1][0] - r[0][1]) / denom, denom * F(.25f)}};
    F qnorm;
    for (int k = 0; k < 4; ++k) {
        q[k] = cases[0][k];
        for (int l = 1; l < 4; ++l)
            q[k] = select(abs(idx - F(float(l))) < F(.5f), cases[l][k], q[k]);
        qnorm = qnorm + q[k] * q[k];
    }
    qnorm = sqrt(qnorm);
    for (int k = 0; k < 4; ++k)
        p.x[3 + k] = q[k] / qnorm;
    p.valid = p.valid & (norm(jet3(j, 1)) <= F(c.max_velocity)) &
              (norm(a) <= F(c.max_acceleration)) & (norm(jerk) <= F(c.max_jerk)) &
              (norm(omega) <= F(c.max_omega)) & (norm(alpha) <= F(c.max_alpha)) &
              (thrust <= F(c.max_thrust));
    return p;
}
} // namespace rrtpff::batch_df
