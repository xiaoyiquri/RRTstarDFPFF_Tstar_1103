#include "simd/physical_batch.h"
#include "simd/simd_backend.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>
namespace rrtpff {
#ifdef RRTPFF_HAS_AVX2
void physicalBatchAVX2(const JetBatch &, const PhysicalBatchConfig &, int, unsigned,
                       const SphereGeometry &, PhysicalBatch &);
void curveBatchAVX2(const BernsteinCurve &, const float *, const PhysicalBatchConfig &, int,
                    unsigned, const SphereGeometry &, PhysicalBatch &);
void stateBatchAVX2(const PhysicalBatchConfig &, unsigned, const SphereGeometry &, PhysicalBatch &);
#endif
namespace {
std::size_t buildNode(SphereGeometry &g, std::size_t begin, std::size_t end) {
    SphereTreeNode n;
    n.begin = begin;
    n.end = end;
    n.lower.fill(std::numeric_limits<float>::infinity());
    n.upper.fill(-std::numeric_limits<float>::infinity());
    for (std::size_t i = begin; i < end; ++i)
        for (int k = 0; k < g.dimension; ++k) {
            float radius = g.spheres[i].radius + g.robot_radius;
            n.lower[k] =
                std::min(n.lower[k], std::nextafter(g.spheres[i].center[k] - radius,
                                                    -std::numeric_limits<float>::infinity()));
            n.upper[k] =
                std::max(n.upper[k], std::nextafter(g.spheres[i].center[k] + radius,
                                                    std::numeric_limits<float>::infinity()));
        }
    const auto id = g.nodes.size();
    g.nodes.push_back(n);
    if (end - begin > 8) {
        int axis = 0;
        for (int k = 1; k < g.dimension; ++k)
            if (n.upper[k] - n.lower[k] > n.upper[axis] - n.lower[axis])
                axis = k;
        auto mid = begin + (end - begin) / 2;
        std::nth_element(
            g.spheres.begin() + begin, g.spheres.begin() + mid, g.spheres.begin() + end,
            [axis](const Sphere &a, const Sphere &b) { return a.center[axis] < b.center[axis]; });
        auto left = buildNode(g, begin, mid), right = buildNode(g, mid, end);
        g.nodes[id].left = left;
        g.nodes[id].right = right;
    }
    return id;
}
void requireAVX2() {
    if (!simdRuntimeAVX2())
        throw std::runtime_error(
            "float8 physical validation requires AVX2; scalar validation is disabled");
}
} // namespace
void SphereGeometry::build() {
    if ((dimension != 2 && dimension != 3) || !std::isfinite(robot_radius) || robot_radius <= 0)
        throw std::invalid_argument("Invalid sphere robot");
    for (const auto &s : spheres) {
        if (!std::isfinite(s.radius) || s.radius <= 0)
            throw std::invalid_argument("Invalid obstacle radius");
        for (int k = 0; k < dimension; ++k)
            if (!std::isfinite(s.center[k]))
                throw std::invalid_argument("Invalid sphere center");
    }
    nodes.clear();
    if (!spheres.empty())
        buildNode(*this, 0, spheres.size());
}
void physicalBatch(const JetBatch &j, const PhysicalBatchConfig &c, int branch, unsigned active,
                   const SphereGeometry &g, PhysicalBatch &out) {
    requireAVX2();
    if (branch != 1 && branch != -1)
        throw std::invalid_argument("Invalid velocity branch");
#ifdef RRTPFF_HAS_AVX2
    physicalBatchAVX2(j, c, branch, active & 255, g, out);
#endif
}
void curveBatch(const BernsteinCurve &curve, const float *times, const PhysicalBatchConfig &c,
                int branch, unsigned active, const SphereGeometry &g, PhysicalBatch &out) {
    requireAVX2();
    if ((branch != 1 && branch != -1) || curve.dim < 2 || curve.dim > 4 ||
        (curve.order != 2 && curve.order != 4) || curve.derivative_order < 0 ||
        curve.derivative_order > 4)
        throw std::invalid_argument("Invalid SIMD curve");
#ifdef RRTPFF_HAS_AVX2
    curveBatchAVX2(curve, times, c, branch, active & 255, g, out);
#endif
}
void stateBatch(const PhysicalBatchConfig &c, unsigned active, const SphereGeometry &g,
                PhysicalBatch &out) {
    requireAVX2();
#ifdef RRTPFF_HAS_AVX2
    stateBatchAVX2(c, active & 255, g, out);
#endif
}
} // namespace rrtpff
