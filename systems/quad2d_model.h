#pragma once
#include "df/quad2d_df.h"
#include "dynobench/quad2d_adapter.h"
#include <variant>
namespace rrtpff {

    class Quad2DModel {

        public:

        static constexpr int FlatDim=2,FlatOrder=4;
        using FlatState=Quad2DDF::FlatState;
        using FlatOutput=FlatState::Output;
        using FlatTrajectory=Quad2DDF::FlatTrajectory;
        using Metadata=typename FlatTrajectory::MetadataType;
        using PhysicalState=Quad2DDF::PhysicalState;
        using PhysicalControl=Quad2DDF::PhysicalControl;

        using PhysicalTrajectory=rrtpff::PhysicalTrajectory<PhysicalState,PhysicalControl>;

        Quad2DModel(FlatState start,FlatOutput goal,FlatOutput lower,FlatOutput upper,Quad2DAdapter adapter= {
        }
        ):start_(start),goal_(goal),lower_(lower),upper_(upper),adapter_(adapter) {
        }

        const FlatState& start()const {
            return start_;
        }
        Metadata initialMetadata() const { return {}; }
        bool prepareEdge(const FlatState&, const Metadata&, FlatTrajectory&) const {
            return true;
        }
        Metadata terminalMetadata(const FlatTrajectory& trajectory) const {
            return trajectory.metadata;
        }
        const FlatOutput& goalOutput()const {
            return goal_;
        }
        const FlatOutput& lowerOutput()const {
            return lower_;
        }
        const FlatOutput& upperOutput()const {
            return upper_;
        }
        FlatOutput limitExtension(const FlatState&s,const FlatOutput&q,double r)const {
            auto d=q-s.output();
            double n=d.norm();
            return n>r?s.output()+r*d/n:q;
        }
        double distanceToOutput(const FlatState&s,const FlatOutput&q)const {
            return(s.output()-q).norm();
        }
        bool initialStateValid()const {
            return start_.finite();
        }
        bool goalOutputValid()const {
            return goal_.allFinite();
        }
        bool isGoal(const FlatState&s,double tol)const {
            return distanceToOutput(s,goal_)<=tol;
        }
        double solutionLowerBound(const FlatState&)const {
            return 0;
        }
        bool trajectoryValid(const FlatTrajectory&tr,PlanResult&)const {
            return Quad2DDF::feasible(tr);
        }
        bool dfFeasible(const FlatTrajectory&tr)const {
            return Quad2DDF::feasible(tr);
        }
        bool physicalTrajectoryValid(const FlatTrajectory&tr)const {
            return Quad2DDF::feasible(tr);
        }
        bool collisionFree(const FlatTrajectory&)const {
            return true;
        }

        private:
        FlatState start_;
        FlatOutput goal_,lower_,upper_;
        Quad2DAdapter adapter_;

    };

}
