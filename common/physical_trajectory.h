#pragma once
#include <vector>

namespace rrtpff {

    template<class StateT,class ControlT>
    struct PhysicalTrajectory {

        using State=StateT;
        using Control=ControlT;

        double T {
            0
        };

        std::vector<State> states;

        std::vector<Control> controls;

        bool valid {
            false
        };

    };

}
