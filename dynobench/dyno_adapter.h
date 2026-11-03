#pragma once
#include <Eigen/Core>
#include <string>
#include <vector>

namespace rrtpff {

    // Dependency-neutral seam. A concrete adapter may wrap DynoBench, but core and
    // PFF only see this contract. `collisionCheck` returns true when collision-free.
    template<class StateT,class ControlT>
    class DynoAdapter {

        public:

        using State=StateT;
        using Control=ControlT;

        virtual ~DynoAdapter()=default;

        virtual bool stateValid(const State&) const=0;

        virtual bool controlValid(const Control&) const=0;

        virtual bool collisionCheck(const State&) const=0;

        virtual bool step(const State&,const Control&,double,State&) const=0;

        virtual bool checkTrajectory(const std::vector<State>&,const std::vector<Control>&,double) const=0;

    };

}
