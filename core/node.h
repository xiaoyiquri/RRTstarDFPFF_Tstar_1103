//实现了 RRT 树的节点结构体 RRTNode，
//用于在树结构中存储状态、路径代价、父子拓扑关系以及链接轨迹等关键信息 
#pragma once
#include "common/flat_trajectory.h"
#include <algorithm>
#include <variant>
#include <vector>

namespace rrtpff {
template <int Dim, int Order, typename MetadataT = std::monostate, class Real = Scalar>
struct RRTNode {
    using State = FlatState<Dim, Order, Real>;
    using Trajectory = FlatTrajectory<Dim, Order, MetadataT, Real>;
    using MetadataType = MetadataT;
    using Metadata = MetadataT;

    State state{};
    Real cost{0};
    RRTNode* parent{nullptr};
    std::vector<RRTNode*> children;
    std::size_t id{0};
    Trajectory incoming{};
    MetadataT metadata{};

    void addChild(RRTNode* child) {
        if (child && std::find(children.begin(), children.end(), child) == children.end())
            children.push_back(child);
    }

    void removeChild(RRTNode* child) {
        children.erase(std::remove(children.begin(), children.end(), child),
                       children.end());
    }
};
}  // namespace rrtpff
