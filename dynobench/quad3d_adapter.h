#pragma once
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cmath>
#include <vector>
namespace rrtpff {

    struct Quad3DAdapter {

        using State=Eigen::Matrix<double,13,1>;
        using Control=Eigen::Matrix<double,4,1>;

        double mass=.034,gravity=9.81,max_motor=1.3,arm_length=.046,t2t=.006,dt=.01;
        Eigen::Vector3d inertia {
            16.571710e-6,16.655602e-6,29.261652e-6
        };

        bool stateValid(const State& x) const {
            if(!x.allFinite())return false;
            Eigen::Quaterniond q(x(6),x(3),x(4),x(5));
            return std::abs(q.norm()-1)<1e-5;
        }

        bool controlValid(const Control& u) const {
            return u.allFinite()&&(u.array()>=0).all()&&(u.array()<=max_motor).all();
        }

        bool collisionCheck(const State&) const {
            return true;
        }

        bool step(const State& x,const Control& u,double h,State& n) const {
            if(h<=0||!stateValid(x)||!controlValid(u))return false;
            n=x;
            Eigen::Quaterniond q(x(6),x(3),x(4),x(5));
            q.normalize();
            const double nominal=mass*gravity/4.;
            Eigen::Vector4d w;
            w<<nominal*u.sum(),0,0,0;
            Eigen::Vector3d thrust=q.toRotationMatrix()*Eigen::Vector3d(0,0,w(0));
            n.head<3>()+=h*x.segment<3>(7);
            n.segment<3>(7)+=h*(thrust/mass+Eigen::Vector3d(0,0,-gravity));
            q.normalize();
            n.segment<4>(3)<<q.x(),q.y(),q.z(),q.w();
            return stateValid(n);
        }

        bool checkTrajectory(const std::vector<State>& states,const std::vector<Control>& controls,double h) const {
            if(states.empty()||controls.size()+1!=states.size())return false;
            for(std::size_t i=0;i<controls.size();++i) {
                State n;
                if(!step(states[i],controls[i],h,n)||(n-states[i+1]).norm()>1e-3)return false;
            }
            return true;
        }

    };

}
