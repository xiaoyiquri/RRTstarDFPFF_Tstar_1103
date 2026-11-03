#pragma once
#include "common/timer.h"
#include "common/types.h"
#include <limits>
#include <stdexcept>
#include <vector>
namespace rrtpff {
struct ShortcutConfig{
    bool enabled{true};
    std::string mode{"flask"},time_mode{"original_duration"};
};
template<class Trajectory> struct ShortcutResult{
    std::vector<Trajectory> path;std::size_t attempts{},accepted{};double seconds{};
};
template<class Model,class Solver> ShortcutResult<typename Model::FlatTrajectory> shortcutPath(
    const std::vector<typename Model::FlatTrajectory>& raw,const Model& model,const Solver& solver,
    const ShortcutConfig& cfg,PlanResult& counters) {
    using Tr=typename Model::FlatTrajectory;using Real=typename Model::FlatOutput::Scalar;
    ShortcutResult<Tr> result;result.path=raw;
    if(!cfg.enabled)return result;
    if((cfg.mode!="flask"&&cfg.mode!="cost_decreasing")||
       (cfg.time_mode!="original_duration"&&cfg.time_mode!="optimal"))
        throw std::invalid_argument("Unknown shortcut policy");
    Timer timer;
    for(std::size_t i=0;i+1<result.path.size();++i){
        for(std::size_t end=result.path.size();end>=i+2;--end){
            ++result.attempts;Real duration=0,cost=0;
            for(std::size_t k=i;k<end;++k){duration+=result.path[k].T;cost+=result.path[k].cost;}
            const auto& first=result.path[i];const auto& last=result.path[end-1];
            const auto target=Solver::Constraint::full(last.end);
            std::vector<typename Solver::Result> choices;
            if(cfg.time_mode=="original_duration")choices.push_back(solver.fixedTime(first.start,target,duration));
            else choices=solver.candidates(first.start,target);
            bool accepted=false;
            for(auto& edge:choices){
                if(!edge.success||!solver.materialize(first.start,edge))continue;
                if(!model.prepareEdge(first.start,first.metadata,edge.trajectory)||
                   !model.terminalMetadataMatches(edge.trajectory,model.terminalMetadata(last)))continue;
                const Real tol=Real(16)*std::numeric_limits<Real>::epsilon()*std::max(Real(1),std::abs(cost));
                if(cfg.mode=="cost_decreasing" && !(edge.cost<cost-tol))continue;
                if(!model.trajectoryValid(edge.trajectory,counters))continue;
                auto pos=result.path.erase(result.path.begin()+i,result.path.begin()+end);
                result.path.insert(pos,std::move(edge.trajectory));
                ++result.accepted;accepted=true;break;
            }
            if(accepted)break;
        }
    }
    result.seconds=timer.seconds();return result;
}
}
