#include "pff/pff.h"
#include "tests/test_support.h"
using namespace rrtpff;

template<int Dim,int Order> void checkEndpointMasks() {
    using Solver=PFFSolver<Dim,Order>;
    using State=typename Solver::State;
    using Weight=typename Solver::Weight;
    State start,goal;
    for(int i=0;i<Dim*Order;++i) {start.values(i)=.1*std::sin(i+.3);
goal.values(i)=.4*std::cos(i+.1);
}
    Weight R=Weight::Identity();
R(0,0)=2;
R(0,1)=R(1,0)=.2;
    Solver solver(R);
    for(int mode=0;mode<3;++mode) {
        auto c=TerminalConstraint<Dim,Order>::output(goal.output());
        if(mode==1) {c.target=goal;
c.fixed[Dim]=true;
c.fixed[Dim*Order-1]=true;
}
        if(mode==2) c=TerminalConstraint<Dim,Order>::full(goal);
        auto r=solver.fixedTime(start,c,1.7);
        require(r.success,"PFF fixed-time solve");
        require((r.trajectory.evaluate(0).values-start.values).norm()<1e-10,"Initial state");
        for(int i=0;i<Dim*Order;++i) {
            if(c.fixed[i]) close(r.terminal_state.values(i),c.target.values(i),1e-7,"Fixed endpoint");
            else close(r.terminal_costate(i),0,1e-12,"Free terminal costate");
        }
        const auto system=IntegratorChain<Dim,Order>::system();
        typename State::Vector integrated=start.values;
        const double dt=r.T/1000;
        auto derivative=[&](const typename State::Vector& x,double t) {
            return (system.A*x+system.B*r.trajectory.control(t)).eval();
        };
        for(int k=0;k<1000;++k) {
            const double t=k*dt;
            const auto k1=derivative(integrated,t);
            const auto k2=derivative((integrated+.5*dt*k1).eval(),t+.5*dt);
            const auto k3=derivative((integrated+.5*dt*k2).eval(),t+.5*dt);
            const auto k4=derivative((integrated+dt*k3).eval(),t+dt);
            integrated+=dt*(k1+2*k2+2*k3+k4)/6;
        }
        require((integrated-r.terminal_state.values).norm()<1e-7,"Numerical dynamics integration");
        PFFCost<Dim> cost;
cost.R=R;
        double integral=0;
        for(int k=0;k<=2000;++k) {
            const double t=r.T*k/2000;
            integral+=(k==0||k==2000?1:k%2?4:2)*cost.runningCost(r.trajectory.control(t),1);
        }
        integral*=r.T/6000;
        close(r.cost,integral,1e-7,"Integrated running cost");
        close(r.cost,cost.totalCost(r.T,solver.energyCoefficients(start,c)),1e-7,"Laurent cost polynomial");
    }
}
int main() {return runTest([] {
    checkEndpointMasks<2,2>();checkEndpointMasks<2,4>();checkEndpointMasks<4,4>();
    FlatState<2,2> x;x.values<<.2,-.1,.5,-.4;
    Eigen::Vector2d y;y<<1.2,.8;
    PFFSolver<2,2> solver;
    const auto c=TerminalConstraint<2,2>::output(y);
    const auto r=solver.fixedTime(x,c,2);
    const Eigen::Vector2d d=x.output()-y,v=x.derivative(1);
    const double old_cost=2+3*v.squaredNorm()/2+6*v.dot(d)/4+3*d.squaredNorm()/8;
    close(r.cost,old_cost,1e-12,"Legacy DI FreeVel cost");
    const Eigen::Vector2d q=d+2*v;
    require((r.terminal_state.derivative(1)-(v-3*q/4)).norm()<1e-12,"Legacy DI free terminal velocity");
    const auto optimum=solver.solve(x,c);
    require(optimum.success&&optimum.T>.001&&optimum.T<5,"Interior free-final-time optimum");
    close(solver.hamiltonian(optimum),0,1e-7,"H(T)=0");
    const auto roots=realRootsInInterval({-2,5,-4,1},.01,3);
    require(roots.size()==2,"Even multiplicity root retained");
    close(roots[0],1,1e-8,"Double root");close(roots[1],2,1e-8,"Simple root");
    auto nan=x;nan.values(0)=std::numeric_limits<double>::quiet_NaN();
    require(!solver.solve(nan,c).success,"NaN rejected");
    require(!solver.fixedTime(x,c,0).success,"Nonpositive time rejected");
    bool bad_R=false;
    try {Eigen::Matrix2d R=Eigen::Matrix2d::Identity();R(0,0)=-1;PFFSolver<2,2> invalid(R);}
    catch(const std::invalid_argument&) {bad_R=true;}
    require(bad_R,"Nonpositive R rejected");
});
}
