#pragma once
#include <random>
#include <type_traits>
#include <utility>

namespace rrtpff {
template <class M, class = void> struct HasSamplingGuidance : std::false_type {};
template <class M>
struct HasSamplingGuidance<M,
                           std::void_t<decltype(std::declval<const M &>().samplingRegions()),
                                       decltype(std::declval<const M &>().guidanceProbability())>>
    : std::true_type {};
template <class Model> class Sampler {
  public:
    using Output = typename Model::FlatOutput;
    using Real = typename Output::Scalar;

    Sampler(const Model &model, unsigned seed) : model_(model), random_(seed) {}

    Output sampleFlatOutput(Real goal_bias, bool *sampled_goal = nullptr) {
        const bool goal = unit_(random_) < goal_bias;
        if (sampled_goal != nullptr)
            *sampled_goal = goal;
        if (goal)
            return model_.goalOutput();

        if constexpr (HasSamplingGuidance<Model>::value) {
            const auto &regions = model_.samplingRegions();
            if (!regions.empty() && unit_(random_) < model_.guidanceProbability()) {
                const auto &region = regions[std::uniform_int_distribution<std::size_t>(
                    0, regions.size() - 1)(random_)];
                Output out;
                for (int i = 0; i < Model::FlatDim; ++i)
                    out(i) = region.lower(i) + unit_(random_) * (region.upper(i) - region.lower(i));
                return out;
            }
        }

        Output out;
        for (int i = 0; i < Model::FlatDim; ++i)
            out(i) = model_.lowerOutput()(i) +
                     unit_(random_) * (model_.upperOutput()(i) - model_.lowerOutput()(i));
        return out;
    }

  private:
    const Model &model_;
    std::mt19937 random_;
    std::uniform_real_distribution<Real> unit_{0, 1};
};
} // namespace rrtpff
