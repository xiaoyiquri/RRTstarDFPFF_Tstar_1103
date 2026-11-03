#pragma once
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace quad3d_test {
struct Row {
    long long id{}, parent{};
    // cost, 16 flat coordinates, duration, 32 polynomial coefficients
    std::array<double, 50> values{};
};
struct Trace {
    // success, node count, PFF evaluations, rewires, valid goal connections
    std::array<long long, 5> counts{};
    std::vector<Row> nodes;
};
inline std::string read(const std::string &path) {
    std::ifstream input(path);
    if (!input)
        throw std::runtime_error("Cannot read trace: " + path);
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}
inline void write(const std::string &path, const std::string &data) {
    std::ofstream output(path);
    output << data;
    output.flush();
    if (!output)
        throw std::runtime_error("Cannot write trace: " + path);
}
inline Trace parse(const std::string &text) {
    Trace trace;
    std::istringstream input(text);
    for (auto &value : trace.counts)
        if (!(input >> value) || value < 0)
            throw std::runtime_error("Malformed trace counts");
    if (trace.counts[0] > 1 || trace.counts[1] < 1 || trace.counts[1] > 1000000)
        throw std::runtime_error("Invalid trace node count/success flag");
    for (long long i = 0; i < trace.counts[1]; ++i) {
        Row row;
        if (!(input >> row.id >> row.parent))
            throw std::runtime_error("Missing trace node " + std::to_string(i));
        for (double &value : row.values)
            if (!(input >> value) || !std::isfinite(value))
                throw std::runtime_error("Missing/nonfinite trace value at node " +
                                         std::to_string(i));
        trace.nodes.push_back(row);
    }
    std::string extra;
    if (input >> extra)
        throw std::runtime_error("Extra data after trace nodes");
    return trace;
}
inline void compare(const Trace &expected, const Trace &actual, const std::string &mode) {
    constexpr const char *count_names[] = {"success", "node_count", "pff_evaluations", "rewires",
                                           "goal_valid"};
    auto mismatch = [&](const std::string &field, auto wanted, auto got) {
        std::ostringstream message;
        message << std::setprecision(17) << mode << ' ' << field << ": expected=" << wanted
                << " actual=" << got;
        throw std::runtime_error(message.str());
    };
    for (std::size_t i = 0; i < expected.counts.size(); ++i)
        if (expected.counts[i] != actual.counts[i])
            mismatch(count_names[i], expected.counts[i], actual.counts[i]);
    if (expected.nodes.size() != actual.nodes.size())
        mismatch("trace_rows", expected.nodes.size(), actual.nodes.size());
    for (std::size_t i = 0; i < expected.nodes.size(); ++i) {
        const auto &a = expected.nodes[i];
        const auto &b = actual.nodes[i];
        const std::string prefix = "node[" + std::to_string(i) + "] ";
        if (a.id != b.id)
            mismatch(prefix + "id", a.id, b.id);
        if (a.parent != b.parent)
            mismatch(prefix + "parent", a.parent, b.parent);
        for (std::size_t j = 0; j < a.values.size(); ++j) {
            // Keep the original numerical tolerance. Discrete structure above
            // must match exactly; nonfinite numbers never compare equal.
            if (std::isfinite(a.values[j]) && std::isfinite(b.values[j]) &&
                std::abs(a.values[j] - b.values[j]) <= 1e-9 * (1 + std::abs(a.values[j])))
                continue;
            const std::string field = j == 0    ? "cost"
                                      : j <= 16 ? "state[" + std::to_string(j - 1) + "]"
                                      : j == 17 ? "duration"
                                                : "coefficient[" + std::to_string(j - 18) + "]";
            mismatch(prefix + field, a.values[j], b.values[j]);
        }
    }
}
inline void checkComparator() {
    Trace expected;
    expected.counts = {1, 2, 10, 0, 1};
    expected.nodes = {Row{0, -1, {}}, Row{1, 0, {}}};
    compare(expected, expected, "self-check");
    auto mustReject = [&](Trace altered) {
        bool rejected = false;
        try {
            compare(expected, altered, "self-check");
        } catch (const std::runtime_error &) {
            rejected = true;
        }
        if (!rejected)
            throw std::runtime_error("Trace comparator accepted a perturbed regression");
    };
    auto altered = expected;
    altered.nodes[1].parent = 1;
    mustReject(altered);
    altered = expected;
    altered.nodes[1].values[1] = .001;
    mustReject(altered);
    altered = expected;
    altered.nodes[1].values[49] = .001;
    mustReject(altered);
    altered = expected;
    ++altered.counts[2];
    mustReject(altered);
    altered = expected;
    altered.nodes.pop_back();
    mustReject(altered);
    altered = expected;
    altered.nodes[1].values[0] = std::numeric_limits<double>::quiet_NaN();
    mustReject(altered);
}
} // namespace quad3d_test
