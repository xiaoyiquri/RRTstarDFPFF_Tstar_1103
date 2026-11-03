#pragma once
#include <array>
#include <cstddef>
#include <limits>
#include <vector>
namespace rrtpff {
enum class BatchRobot { Unicycle, Quad2D, Quad3D };
struct Sphere {
    std::array<float, 3> center{};
    float radius{};
};
struct SphereTreeNode {
    std::array<float, 3> lower{}, upper{};
    std::size_t begin{}, end{}, left{}, right{};
};
struct SphereGeometry {
    float robot_radius{};
    int dimension{2};
    std::vector<Sphere> spheres;
    std::vector<SphereTreeNode> nodes;
    void build();
};
struct PhysicalBatchConfig {
    BatchRobot robot{BatchRobot::Unicycle};
    int state_dim{3}, control_dim{2}, position_dim{2};
    float mass{1}, gravity{9.81f}, inertia2{1e-4f}, arm{.1f};
    std::array<float, 3> inertia{{.1f, .1f, .2f}};
    float max_velocity{4}, max_omega{8};
    float max_acceleration{std::numeric_limits<float>::infinity()};
    float max_jerk{std::numeric_limits<float>::infinity()};
    float max_alpha{std::numeric_limits<float>::infinity()};
    float max_thrust{std::numeric_limits<float>::infinity()}, epsilon{1e-8f};
    std::array<float, 13> state_lower{}, state_upper{};
    std::array<float, 4> control_lower{}, control_upper{};
    std::array<std::array<float, 4>, 4> wrench_inverse{};
};
// SoA: derivative order, flat dimension, 8 different times on ONE edge.
struct JetBatch {
    float values[5][4][8]{};
};
struct PhysicalBatch {
    float states[13][8]{}, controls[4][8]{};
    unsigned df_mask{}, state_mask{}, control_mask{}, collision_mask{};
    unsigned valid_mask{};
    std::size_t sphere_tests{};
};
struct BernsteinCurve {
    int dim{}, order{};
    // Integrator order and physical reconstruction derivative order differ for r=2 quads.
    int derivative_order{};
    float coefficients[5][4][8]{};
};
// These entry points fail explicitly without AVX2; they never validate scalarly.
void physicalBatch(const JetBatch &, const PhysicalBatchConfig &, int branch, unsigned active,
                   const SphereGeometry &, PhysicalBatch &);
void curveBatch(const BernsteinCurve &, const float times[8], const PhysicalBatchConfig &,
                int branch, unsigned active, const SphereGeometry &, PhysicalBatch &);
void stateBatch(const PhysicalBatchConfig &, unsigned active, const SphereGeometry &,
                PhysicalBatch &);
} // namespace rrtpff
