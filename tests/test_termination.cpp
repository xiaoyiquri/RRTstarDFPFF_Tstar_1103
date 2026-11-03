#include "tests/test_support.h"
#include "core/rrtstar.h"
#include "systems/linear_model.h"
using namespace rrtpff;
class GoalModel:public LinearModel{
public:
    explicit GoalModel(Config c):LinearModel(c){goal_.setDerivative(0,c.goal);}
    TerminalConstraint<2,2> goalConstraint()const{return TerminalConstraint<2,2>::full(goal_);}
    bool isGoal(const FlatState& x,double tol)const{return (x.values-goal_.values).norm()<=tol;}
private:FlatState goal_;
};
int main(){return runTest([]{
    PlannerConfig pc;pc.max_nodes=pc.max_iterations=0;pc.timeout=.02;pc.validate();
    PlannerConfigT<float> fp;fp.max_nodes=fp.max_iterations=0;fp.validate();
    LinearModel::Config c;c.start.values<<1,1,0,0;c.goal<<3,1;c.upper<<5,5;
    pc.goal_bias=1;pc.explicit_goal_connection=true;pc.stop_at_first_solution=true;
    pc.rewire_before_first_solution=false;pc.prune=true;pc.timeout=1;
    RRTstar<GoalModel> planner(GoalModel(c),pc);auto result=planner.solve();
    require(result.success&&result.termination_reason=="first_solution","latency exits on first goal");
    require(result.iterations==1&&result.rewire_count==0,"no goal-iteration rewire");
    require(!planner.firstPath().empty()&&planner.firstPath().size()==planner.path().size(),"first path snapshot saved");
    close(result.first_solution_cost,result.cost,1e-8,"first solution cost snapshot");
    pc.stop_at_first_solution=false;pc.explicit_goal_connection=false;pc.goal_bias=0;
    pc.hard_safety_node_cap=2;pc.prune=false;
    RRTstar<GoalModel> safety(GoalModel(c),pc);auto stopped=safety.solve();
    require(stopped.termination_reason=="safety_limit"&&!stopped.completed_normally,"emergency cap abnormal");
    pc.hard_safety_node_cap=0;pc.timeout=.03;
    RRTstar<GoalModel> timed(GoalModel(c),pc);auto timeout=timed.solve();
    require(timeout.iterations>0&&timeout.termination_reason=="timeout","zero normal caps allow timed work");
});}
