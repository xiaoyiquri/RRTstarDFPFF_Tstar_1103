#pragma once
#include <Eigen/Core>
#include <cmath>
#include <vector>
namespace rrtpff {

    struct Quad2DAdapter {

        using State=Eigen::Matrix<double,6,1>;
        using Control=Eigen::Matrix<double,2,1>;

        double mass=.034,gravity=9.81,inertia=1e-4,arm=.1,dt=.01,max_motor=1.3;

        bool stateValid(const State& x) const {
            return x.allFinite();
        }

        bool controlValid(const Control& u) const {
            return u.allFinite()&&(u.array()>=0).all()&&(u.array()<=max_motor).all();
        }

        bool collisionCheck(const State&) const {
            return true;
        }

        bool step(const State& x,const Control& u,double h,State& n) const {
            if(h<=0||!stateValid(x)||!controlValid(u))return false;
            const double f1=mass*gravity*u(0)/2,f2=mass*gravity*u(1)/2,theta=x(2),thrust=f1+f2;
            n=x;
            n(0)+=h*x(3);
            n(1)+=h*x(4);
            n(2)+=h*x(5);
            n(3)+=h*(-thrust*std::sin(theta)/mass);
            n(4)+=h*(thrust*std::cos(theta)/mass-gravity);
            n(5)+=h*arm*(f1-f2)/inertia;
            return n.allFinite();
        }

        bool checkTrajectory(const std::vector<State>& states,const std::vector<Control>& controls,double h) const {
            if(states.empty()||controls.size()+1!=states.size())return false;
            for(std::size_t i=0;i<controls.size();++i) {
                State n;
                if(!step(states[i],controls[i],h,n)||(n-states[i+1]).norm()>1e-4)return false;
            }
            return true;
        }

    };

}
