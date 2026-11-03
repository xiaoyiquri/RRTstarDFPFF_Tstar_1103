#pragma once
#include "df/df_types.h"
#include "common/physical_trajectory.h"

namespace rrtpff {

    template<class Impl>
    class DifferentialFlatness {

        public:

        using FlatState=typename Impl::FlatState;

        using FlatTrajectory=typename Impl::FlatTrajectory;

        using FlatControl=typename Impl::FlatControl;

        using LiftData=typename Impl::LiftData;

        using PhysicalState=typename Impl::PhysicalState;

        using PhysicalControl=typename Impl::PhysicalControl;

        using Metadata=typename FlatTrajectory::MetadataType;

        bool stateToFlat(const PhysicalState& state,
                         const LiftData& lift,
                         FlatState& flat) const {
            return Impl::stateToFlat(state, lift, flat);
        }

        bool flatToPhysicalState(const FlatState& flat,
                                 const FlatControl& control,
                                 const Metadata& metadata,
                                 PhysicalState& state) const {
            return Impl::flatToPhysicalState(flat, control, metadata, state);
        }

        bool flatToPhysicalControl(const FlatState& flat,
                                   const FlatControl& flat_control,
                                   const Metadata& metadata,
                                   PhysicalControl& control) const {
            return Impl::flatToPhysicalControl(flat, flat_control, metadata, control);
        }

        bool trajectoryState(const FlatTrajectory& trajectory,
                             double t,
                             PhysicalState& state) const {
            return Impl::trajectoryState(trajectory, t, state);
        }

        bool trajectoryControl(const FlatTrajectory& trajectory,
                               double t,
                               PhysicalControl& control) const {
            return Impl::trajectoryControl(trajectory, t, control);
        }

        bool feasible(const FlatTrajectory& trajectory) const {
            return Impl::feasible(trajectory);
        }

    };

}
