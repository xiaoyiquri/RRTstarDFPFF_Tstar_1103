#pragma once
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <locale>
#include <stdexcept>

namespace rrtpff {
// Serialize the FINAL retained tree, including every off-path branch. Store
// polynomial coefficients rather than chords or pre-sampled large point arrays.
// Called only after solve(), while the planner and its parent pointers live.
template<int PositionDim, class Tree>
void writeSearchTree(const Tree& tree, const std::filesystem::path& file) {
    static_assert(PositionDim == 2 || PositionDim == 3);
    auto temporary = file;
    temporary += ".tmp";
    try {
        std::ofstream out(temporary);
        out.exceptions(std::ios::failbit | std::ios::badbit);
        out.imbue(std::locale::classic());
        out << std::setprecision(std::numeric_limits<double>::max_digits10);
        out << "{\"schema_version\":1,\"snapshot\":\"final\",\"position_dim\":"
            << PositionDim << ",\"node_count\":" << tree.size() << ",\"nodes\":[\n";
        bool comma = false;
        for (const auto& ptr : tree.nodes()) {
            const auto& node = *ptr;
            if (comma) out << ",\n";
            comma = true;
            out << "{\"id\":" << node.id << ",\"parent_id\":";
            if (node.parent) out << node.parent->id; else out << "null";
            out << ",\"position\":[";
            const auto position = node.state.output();
            for (int axis = 0; axis < PositionDim; ++axis)
                out << (axis ? "," : "") << position(axis);
            out << "],\"cost\":" << node.cost;
            if (node.parent) {
                const auto& edge = node.incoming;
                out << ",\"duration\":" << edge.T << ",\"normalized\":"
                    << (edge.normalized ? "true" : "false") << ",\"coefficients\":[";
                for (int axis = 0; axis < PositionDim; ++axis) {
                    out << (axis ? ",[" : "[");
                    for (int k = 0; k < edge.coefficients.cols(); ++k)
                        out << (k ? "," : "") << edge.coefficients(axis, k);
                    out << ']';
                }
                out << ']';
            }
            out << '}';
        }
        out << "\n]}\n";
        out.close();
        std::filesystem::rename(temporary, file);
    } catch (...) {
        std::error_code ignored;
        std::filesystem::remove(temporary, ignored);
        throw;
    }
}
} // namespace rrtpff
