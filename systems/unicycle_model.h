#pragma once
#include "df/unicycle_df.h"
#include "dynobench/uni_adapter.h"
#include "common/types.h"
#include <cmath>
#include <limits>
#include <variant>
namespace rrtpff {

    class UnicycleModel {

        public:

        static constexpr int FlatDim=2,FlatOrder=2;
        using FlatState=UnicycleDF::FlatState;
        using FlatOutput=FlatState::Output;
        using FlatTrajectory=UnicycleDF::FlatTrajectory;
        using Metadata=typename FlatTrajectory::MetadataType;
        using PhysicalState=UnicycleDF::PhysicalState;
        using PhysicalControl=UnicycleDF::PhysicalControl;

        using PhysicalTrajectory=rrtpff::PhysicalTrajectory<PhysicalState,PhysicalControl>;

        UnicycleModel(FlatState start,
                      FlatOutput goal,
                      FlatOutput lower,
                      FlatOutput upper,
                      UniAdapter adapter = {},
                      UnicycleDF::Config df_config = {},
                      double signed_start_velocity =
                          std::numeric_limits<double>::quiet_NaN())
            : start_(start),
              goal_(goal),
              lower_(lower),
              upper_(upper),
              adapter_(adapter),
              df_config_(df_config),
              signed_start_velocity_(signed_start_velocity) {
        }

        const FlatState& start()const {
            return start_;
        }
        Metadata initialMetadata() const {
            Metadata metadata;
            if (!UnicycleDF::makeMetadata(signed_start_velocity_, metadata,
                                          df_config_)) {
                return {};
            }
            return metadata;
        }

        bool prepareEdge(const FlatState& parent,
                         const Metadata& parent_metadata,
                         FlatTrajectory& edge) const {
            if (!UnicycleDF::validMetadata(parent_metadata, df_config_) ||
                !parent.finite() || !edge.start.finite() ||
                !edge.coefficients.allFinite() ||
                !std::isfinite(edge.T) || edge.T <= 0.0 ||
                !edge.start.values.isApprox(parent.values, 1e-9)) {
                return false;
            }
            const double speed = edge.start.derivative(1).norm();
            const double expected =
                std::abs(parent_metadata.signed_start_velocity);
            if (!std::isfinite(speed) || speed <= df_config_.velocity_epsilon ||
                std::abs(speed - expected) >
                    df_config_.start_tolerance * (1.0 + expected)) {
                return false;
            }
            edge.metadata = parent_metadata;
            return true;
        }

        Metadata terminalMetadata(const FlatTrajectory& edge) const {
            Metadata metadata;
            if (!UnicycleDF::validMetadata(edge.metadata, df_config_) ||
                !edge.end.finite()) {
                return {};
            }
            const double speed = edge.end.derivative(1).norm();
            const double signed_speed =
                edge.metadata.branch == VelocityBranch::Reverse ? -speed : speed;
            if (!UnicycleDF::makeMetadata(signed_speed, metadata, df_config_)) {
                return {};
            }
            return metadata;
        }

        bool terminalMetadataMatches(const FlatTrajectory& edge,
                                     const Metadata& target_metadata) const {
            if (!UnicycleDF::validMetadata(target_metadata, df_config_)) {
                return false;
            }
            const Metadata terminal = terminalMetadata(edge);
            if (!UnicycleDF::validMetadata(terminal, df_config_) ||
                terminal.branch != target_metadata.branch) {
                return false;
            }
            // Edge metadata records its START speed. A fixed-endpoint rewire
            // must compare the reconstructed END speed with the frozen node.
            return std::abs(terminal.signed_start_velocity -
                            target_metadata.signed_start_velocity) <=
                df_config_.start_tolerance *
                    (1.0 + std::abs(target_metadata.signed_start_velocity));
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

        FlatOutput limitExtension(const FlatState& s,const FlatOutput& q,double r)const {
            auto d=q-s.output();
            double n=d.norm();
            return n>r?s.output()+r*d/n:q;
        }

        double distanceToOutput(const FlatState& s,const FlatOutput& q)const {
            return(s.output()-q).norm();
        }
        bool initialStateValid()const {
            const Metadata metadata = initialMetadata();
            return start_.finite() &&
                UnicycleDF::validMetadata(metadata, df_config_) &&
                start_.derivative(1).norm() > df_config_.velocity_epsilon &&
                start_.derivative(1).norm() <= df_config_.max_velocity + 1e-12 &&
                std::abs(start_.derivative(1).norm() -
                         std::abs(metadata.signed_start_velocity)) <=
                    df_config_.start_tolerance *
                        (1.0 + std::abs(metadata.signed_start_velocity));
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

        bool trajectoryValid(const FlatTrajectory& tr,PlanResult&)const {
            return UnicycleDF::feasible(tr, df_config_);
        }

        bool dfFeasible(const FlatTrajectory& tr)const {
            return UnicycleDF::feasible(tr, df_config_);
        }
        bool physicalTrajectoryValid(const FlatTrajectory& tr)const {
            return UnicycleDF::feasible(tr, df_config_);
        }
        bool collisionFree(const FlatTrajectory&)const {
            return true;
        }

        private:
        FlatState start_;
        FlatOutput goal_,lower_,upper_;
        UniAdapter adapter_;
        UnicycleDF::Config df_config_;
        double signed_start_velocity_;

    };

}
