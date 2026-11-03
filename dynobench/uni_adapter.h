#pragma once
#include "dynobench/dyno_adapter.h"
#include <Eigen/Core>
#include <cmath>
#include <limits>

namespace rrtpff {

    struct UniAdapter {

        using State=Eigen::Matrix<double,3,1>;
        using Control=Eigen::Matrix<double,2,1>;

        Eigen::Vector3d lower {
            -std::numeric_limits<double>::infinity(),-std::numeric_limits<double>::infinity(),-std::numeric_limits<double>::infinity()
        };

        Eigen::Vector3d upper {
            std::numeric_limits<double>::infinity(),std::numeric_limits<double>::infinity(),std::numeric_limits<double>::infinity()
        };

        double max_velocity=.5,max_omega=.5;

        bool stateValid(const State& x) const {
            return x.allFinite()&&(x.array()>=lower.array()).all()&&(x.array()<=upper.array()).all();
        }

        bool controlValid(const Control& u) const {
            return u.allFinite()&&std::abs(u(0))<=max_velocity&&std::abs(u(1))<=max_omega;
        }

        bool collisionCheck(const State&) const {
            return true;
        }

        bool step(const State& x,const Control& u,double dt,State& next) const {
            if(!stateValid(x)||!controlValid(u)||dt<=0)return false;
            next=x;
            next(0)+=dt*u(0)*std::cos(x(2));
            next(1)+=dt*u(0)*std::sin(x(2));
            next(2)=std::atan2(std::sin(x(2)+dt*u(1)),std::cos(x(2)+dt*u(1)));
            return stateValid(next);
        }

        bool checkTrajectory(const std::vector<State>& states,const std::vector<Control>& controls,double dt) const {
            if(states.empty()||controls.size()+1!=states.size())return false;
            for(std::size_t i=0;i<controls.size();++i) {
                State next;
                if(!step(states[i],controls[i],dt,next)||(next-states[i+1]).norm()>1e-5)return false;
            }
            return true;
        }

    };

}
