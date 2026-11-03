#pragma once
#include "common/physical_trajectory.h"
#include <type_traits>
#include <utility>

namespace rrtpff {

    // Composition point for one system: flat representation + DF + physical
    // constraints + environment + DynoBench adapter. It contains no RRT logic.
    template<class DF,class Adapter,class Environment>
    class SystemModel {

        public:

        using FlatState=typename DF::FlatState;
        using FlatTrajectory=typename DF::FlatTrajectory;

        using PhysicalState=typename DF::PhysicalState;
        using PhysicalControl=typename DF::PhysicalControl;
        using LiftData=typename DF::LiftData;
        using FlatControl=typename DF::FlatControl;
        using Metadata=typename FlatTrajectory::MetadataType;

        using PhysicalTrajectory=rrtpff::PhysicalTrajectory<PhysicalState,PhysicalControl>;

        using FlatOutput=typename FlatState::Output;
        static constexpr int FlatDim=FlatState::OutputDim;
        static constexpr int FlatOrder=FlatState::IntegratorOrder;

        SystemModel(DF df,Adapter adapter,Environment environment):df_(std::move(df)),adapter_(std::move(adapter)),environment_(std::move(environment)) {
        }

        bool stateToFlat(const PhysicalState& state,
                         const LiftData& lift,
                         FlatState& flat) const {
            return df_.stateToFlat(state, lift, flat);
        }

        Metadata initialMetadata() const {
            return {};
        }

        bool prepareEdge(const FlatState&, const Metadata&,
                         FlatTrajectory&) const {
            return true;
        }

        Metadata terminalMetadata(const FlatTrajectory& trajectory) const {
            return trajectory.metadata;
        }

        bool flatToPhysicalState(const FlatState& flat,
                                 const FlatControl& control,
                                 const Metadata& metadata,
                                 PhysicalState& state) const {
            return df_.flatToPhysicalState(flat, control, metadata, state);
        }

        bool flatToPhysicalControl(const FlatState& flat,
                                   const FlatControl& control,
                                   const Metadata& metadata,
                                   PhysicalControl& physical_control) const {
            return df_.flatToPhysicalControl(flat, control, metadata,
                                             physical_control);
        }

        bool trajectoryValid(const FlatTrajectory& trajectory) const {
            return df_.feasible(trajectory);
        }

        double flatDistance(const FlatState& a,const FlatState& b) const {
            return (a.output()-b.output()).norm();
        }

        const DF& df() const {
            return df_;
        }
        const Adapter& adapter() const {
            return adapter_;
        }
        const Environment& environment()const {
            return environment_;
        }

        private:
        DF df_;
        Adapter adapter_;
        Environment environment_;

    };

}
