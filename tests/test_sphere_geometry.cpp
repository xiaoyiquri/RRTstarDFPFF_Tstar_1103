#include "dynobench/data_adapter.h"
#include "simd/physical_batch.h"
#include "tests/test_support.h"
#include <filesystem>
#include <fstream>
#include <random>
using namespace rrtpff;
int main() {
    return runTest([] {
        std::mt19937 rng(928);
        std::uniform_real_distribution<float> random(-5, 5);
        for (int dim : {2, 3}) {
            SphereGeometry geometry;
            geometry.dimension = dim;
            geometry.robot_radius = .125f;
            for (int i = 0; i < 257; ++i)
                geometry.spheres.push_back(
                    {{random(rng), random(rng), dim == 3 ? random(rng) : 0.f}, .2f});
            geometry.build();
            PhysicalBatchConfig c;
            c.position_dim = dim;
            c.state_dim = dim;
            c.control_dim = 0;
            c.state_lower.fill(-100);
            c.state_upper.fill(100);
            for (int i = 0; i < 500; ++i) {
                PhysicalBatch out;
                unsigned active = unsigned(i) % 256, expected = 0;
                for (int lane = 0; lane < 8; ++lane) {
                    for (int k = 0; k < dim; ++k)
                        out.states[k][lane] = random(rng);
                    // Independent brute-force geometry reference, used only in tests.
                    for (const auto &sphere : geometry.spheres) {
                        double squared = 0;
                        for (int k = 0; k < dim; ++k)
                            squared += std::pow(double(out.states[k][lane]) - sphere.center[k], 2);
                        double r = double(sphere.radius) + geometry.robot_radius;
                        if (squared <= r * r) {
                            expected |= 1u << lane;
                            break;
                        }
                    }
                }
                stateBatch(c, active, geometry, out);
                require(out.collision_mask == (expected & active),
                        "BVH cannot omit active colliding spheres");
                require(out.valid_mask == (active & ~expected), "valid geometry mask");
            }
        }
        // YAML mapping and actual conservative sphere cover, independent of FCL.
        const std::filesystem::path dir = "geometry_test_data";
        std::filesystem::create_directories(dir);
        {
            std::ofstream f(dir / "unicycle1_test.yaml");
            f << "dynamics: unicycle1\nmax_vel: 1\nmin_vel: -0.5\nmax_angular_vel: "
                 "1.5\nmin_angular_vel: -1.5\ndt: .1\n";
            std::ofstream e(dir / "environment.yaml");
            e << "environment:\n  min: [-5, -5]\n  max: [5, 5]\n  obstacles:\n    - {type: box, "
                 "center: [0, 0], size: [1, 1]}\nrobots:\n  - {type: unicycle1_test, start: [-2, "
                 "0, 0], goal: [2, 0, 0]}\n";
        }
        PaperDataAdapter adapter((dir / "environment.yaml").string(),
                                 (dir / "unicycle1_test.yaml").string(), .125, .1, 1e-5);
        require(adapter.geometry().spheres.size() == 100, "sphere cover resolution");
        close(adapter.problem().lower_control(0), -.5, 0, "signed velocity bound mapping");
        const auto &g = adapter.geometry();
        for (int x = 0; x <= 40; ++x)
            for (int y = 0; y <= 40; ++y) {
                double px = -.5 + x / 40., py = -.5 + y / 40.;
                bool covered = false;
                for (const auto &s : g.spheres)
                    if (std::hypot(px - s.center[0], py - s.center[1]) <= s.radius) {
                        covered = true;
                        break;
                    }
                require(covered, "box must be covered including boundary");
            }
    });
}
