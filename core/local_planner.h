#pragma once
#include "common/physical_trajectory.h"
#include "pff/pff.h"
#include <utility>

namespace rrtpff {

    template<class Model>
    struct LocalPlanResult {

        using Solver=PFFSolver<Model::FlatDim,Model::FlatOrder,
                               typename Model::Metadata>;
        using FlatTrajectory=typename Model::FlatTrajectory;

        using PhysicalTrajectory=typename Model::PhysicalTrajectory;

        bool pff_success {
            false
        }
        ,df_success {
            false
        }
        ,physical_valid {
            false
        }
        ,collision_free {
            false
        };

        double T {
            0
        }
        ,cost {
            0
        };
        typename Solver::Result pff{};
        FlatTrajectory flat_trajectory{};

    };

    // Scalar local planner. SIMD candidate evaluation is intentionally injected
    // later; this class remains the single semantic reference implementation.
    template<class Model>
    class LocalPlanner {

        public:

        using Solver=PFFSolver<Model::FlatDim,Model::FlatOrder,
                               typename Model::Metadata>;
        using State=typename Model::FlatState;
        using Constraint=typename Solver::Constraint;

        LocalPlanner(const Model& model,Solver solver):model_(model),solver_(std::move(solver)) {
        }

        LocalPlanResult<Model> connectOutput(const State& parent,const typename Model::FlatOutput& output) const {

            LocalPlanResult<Model> out;
            out.pff=solver_.solve(parent,Constraint::output(output));
            out.pff_success=out.pff.success;
            if(!out.pff_success)return out;
            out.flat_trajectory=out.pff.trajectory;
            out.T=out.pff.T;
            out.cost=out.pff.cost;
            out.df_success=model_.dfFeasible(out.flat_trajectory);
            out.physical_valid=out.df_success&&model_.physicalTrajectoryValid(out.flat_trajectory);
            out.collision_free=out.physical_valid&&model_.collisionFree(out.flat_trajectory);
            return out;

        }

        LocalPlanResult<Model> connectState(const State& parent,const State& target) const {

            LocalPlanResult<Model> out;
            out.pff=solver_.solve(parent,Constraint::full(target));
            out.pff_success=out.pff.success;
            if(!out.pff_success) return out;
            out.flat_trajectory=out.pff.trajectory;
            out.T=out.pff.T;
            out.cost=out.pff.cost;
            out.df_success=model_.dfFeasible(out.flat_trajectory);
            out.physical_valid=out.df_success&&model_.physicalTrajectoryValid(out.flat_trajectory);
            out.collision_free=out.physical_valid&&model_.collisionFree(out.flat_trajectory);
            return out;

        }

        private:
        const Model& model_;
        Solver solver_;

    };

}
