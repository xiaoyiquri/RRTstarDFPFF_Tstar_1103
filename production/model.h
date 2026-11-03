#pragma once
#include "production/float_validator.h"
#include "production/unicycle_speed_certificate.h"
#include "common/boundary_diagnostics.h"
#include "dynobench/data_adapter.h"
#include <Eigen/Geometry>
#include <yaml-cpp/yaml.h>
namespace rrtpff::production {
enum class FloatVelocityBranch{Forward,Reverse};
struct FloatUnicycleMetadata{
    float signed_start_velocity{};
    FloatVelocityBranch branch{FloatVelocityBranch::Forward};
};
template<BatchRobot Robot> class FloatModel {
public:
    static constexpr bool Uni=Robot==BatchRobot::Unicycle,Quad3=Robot==BatchRobot::Quad3D;
    static constexpr int FlatDim=Quad3?4:2,FlatOrder=2,StateDim=Uni?3:(Quad3?13:6);
    using FlatState=rrtpff::FlatState<FlatDim,2,float>;
    using FlatOutput=typename FlatState::Output;
    using Metadata=std::conditional_t<Uni,FloatUnicycleMetadata,std::monostate>;
    using FlatTrajectory=rrtpff::FlatTrajectory<FlatDim,2,Metadata,float>;
    using Solver=R2FloatSolver<FlatDim,Metadata>;
    using BatchSolver=R2FloatBatch<FlatDim,Metadata>;
    using PhysicalState=Eigen::Matrix<float,StateDim,1>;
    using Constraint=typename Solver::Constraint;
    FloatModel(const PaperDataAdapter& adapter,const YAML::Node& y):geometry_(&adapter.geometry()) {
        const auto& d=adapter.problem();
        if(d.start.size()!=StateDim)throw std::invalid_argument("Robot state dimension mismatch");
        requested_start_=d.start.cast<float>();requested_goal_=d.goal.cast<float>();
        max_step_=y["validation"]["max_step"].as<float>();max_samples_=y["validation"]["max_samples"].as<std::size_t>();
        endpoint_tolerance_=y["validation"]["endpoint_tolerance"].as<float>();
        if(!(max_step_>0.f)||!std::isfinite(max_step_)||max_samples_<2||!(endpoint_tolerance_>0.f))
            throw std::invalid_argument("Invalid float validation configuration");
        lower_.setZero();upper_.setZero();
        constexpr int PD=Quad3?3:2;
        lower_.template head<PD>()=d.lower_position.cast<float>();
        upper_.template head<PD>()=d.upper_position.cast<float>();
        batch_.robot=Robot;batch_.state_dim=StateDim;batch_.position_dim=PD;
        batch_.control_dim=Uni||!Quad3?2:4;
        batch_.max_velocity=float(d.max_velocity);batch_.max_omega=float(d.max_omega);
        batch_.epsilon=Uni?1e-6f:1e-8f;
        for(int k=0;k<StateDim;++k){batch_.state_lower[k]=float(d.lower_state(k));batch_.state_upper[k]=float(d.upper_state(k));}
        for(int k=0;k<batch_.control_dim;++k){batch_.control_lower[k]=float(d.lower_control(k));batch_.control_upper[k]=float(d.upper_control(k));}
        if constexpr(!Uni){
            batch_.mass=float(d.mass);batch_.gravity=float(d.gravity);batch_.arm=float(d.arm_length);
            batch_.inertia2=float(d.inertia_2d);
            for(int k=0;k<3;++k)batch_.inertia[k]=float(d.inertia_3d(k));
        }
        if constexpr(Uni){
            const float a=y["lifting"]["start"]["signed_velocity"].as<float>(),b=y["lifting"]["goal"]["signed_velocity"].as<float>();
            if(a*b<=0.f||std::abs(a)<=batch_.epsilon||std::abs(b)<=batch_.epsilon ||
               a<batch_.control_lower[0]||a>batch_.control_upper[0]||
               b<batch_.control_lower[0]||b>batch_.control_upper[0])
                throw std::invalid_argument("Unicycle requires same nonzero legal signed branch at both endpoints");
            branch_=a<0.f?-1:1;initial_.signed_start_velocity=a;
            initial_.branch=branch_<0?FloatVelocityBranch::Reverse:FloatVelocityBranch::Forward;
            start_=lift(requested_start_,a);goal_=lift(requested_goal_,b);
        }else if constexpr(Quad3){
            Eigen::Matrix4f actuation=Eigen::Matrix4f::Zero();
            if(d.motor_control){
                const float a=.707106781f*batch_.arm,k=float(d.thrust_to_torque);
                actuation<<1.f,1.f,1.f,1.f,-a,-a,a,a,-a,a,a,-a,-k,k,-k,k;
                actuation*=batch_.mass*batch_.gravity/4.f;
            }else actuation.diagonal()<<batch_.mass*batch_.gravity,20.f,20.f,20.f;
            const Eigen::Matrix4f inverse=actuation.inverse();
            if(!inverse.allFinite())throw std::invalid_argument("Invalid float actuation map");
            for(int k=0;k<4;++k)for(int j=0;j<4;++j)batch_.wrench_inverse[k][j]=inverse(k,j);
            batch_.max_thrust=batch_.mass*batch_.gravity*float(d.max_motor);
            lower_(3)=y["sampling"]["yaw_lower"].as<float>(0.f);
            upper_(3)=y["sampling"]["yaw_upper"].as<float>(0.f);
            start_=lift(requested_start_,0.f);goal_=lift(requested_goal_,0.f);
        }else{
            start_=lift(requested_start_,0.f);goal_=lift(requested_goal_,0.f);
        }
        // Cold input conversion ends here. No double model/DF data enters search.
        if(y["df_limits"]){
            batch_.max_acceleration=y["df_limits"]["max_acceleration"].as<float>(batch_.max_acceleration);
            batch_.max_alpha=y["df_limits"]["max_angular_acceleration"].as<float>(batch_.max_alpha);
        }
        if(!start_.finite()||!goal_.finite()||!lower_.allFinite()||!upper_.allFinite())
            throw std::invalid_argument("Nonfinite float model");
    }
    const FlatState& start()const{return start_;}
    const FlatOutput goalOutput()const{return goal_.output();}
    const FlatOutput& lowerOutput()const{return lower_;}
    const FlatOutput& upperOutput()const{return upper_;}
    FlatOutput metricWeights()const{return FlatOutput::Ones();}
    FlatOutput metricPeriods()const{return FlatOutput::Zero();}
    float distanceToOutput(const FlatState& s,const FlatOutput& q)const{return (s.output()-q).norm();}
    FlatOutput limitExtension(const FlatState& s,const FlatOutput& q,float range)const {
        FlatOutput d=q-s.output();const float n=d.norm();
        return n>range?(s.output()+d*(range/n)).eval():q;
    }
    Metadata initialMetadata()const{return initial_;}
    Constraint expansionConstraint(const FlatOutput& target)const{
        return (target-goal_.output()).squaredNorm()<1e-12f?Constraint::full(goal_):Constraint::output(target);
    }
    Constraint goalConstraint()const{return Constraint::full(goal_);}
    bool isGoal(const FlatState& s,float tolerance)const{return (s.values-goal_.values).norm()<=tolerance;}
    float solutionLowerBound(const FlatState&)const{return 0.f;}
    bool initialStateValid()const{return stateValid(requested_start_);}
    bool goalOutputValid()const{return stateValid(requested_goal_);}
    bool prepareEdge(const FlatState& parent,const Metadata& metadata,FlatTrajectory& tr)const{
        if(!tr.start.values.isApprox(parent.values,1e-5f))return false;
        if constexpr(Uni){
            if(metadata.branch!=initial_.branch||
               std::abs(parent.derivative(1).norm()-std::abs(metadata.signed_start_velocity))>1e-5f*(1.f+std::abs(metadata.signed_start_velocity)))
                return false;
        }
        tr.metadata=metadata;return true;
    }
    Metadata terminalMetadata(const FlatTrajectory& tr)const{
        if constexpr(Uni)return {float(branch_)*tr.end.derivative(1).norm(),initial_.branch};
        else return {};
    }
    bool terminalMetadataMatches(const FlatTrajectory& tr,const Metadata& metadata)const{
        if constexpr(Uni)return metadata.branch==tr.metadata.branch &&
            std::abs(terminalMetadata(tr).signed_start_velocity-metadata.signed_start_velocity)<1e-5f*(1.f+std::abs(metadata.signed_start_velocity));
        return true;
    }
    bool trajectoryValid(const FlatTrajectory& tr,PlanResult& stats)const{return validate(tr,stats,nullptr);}
    float sampleStep()const{return max_step_;}
    DynoValidation validatePath(const std::vector<FlatTrajectory>& path,DynoSamples& all,PlanResult& stats)const{
        all={};stats.boundary_diagnostics.clear();stats.physical_boundary_match=false;
        if(path.empty())return {false,0,0,0,0,"empty_path"};
        if((path.front().start.values-start_.values).norm()>endpoint_tolerance_||
           (path.back().end.values-goal_.values).norm()>endpoint_tolerance_)
            return {false,0,0,0,0,"flat_endpoint_mismatch"};
        float elapsed=0.f;Eigen::VectorXd previous_end;FlatState previous_flat;bool match=true;
        for(std::size_t e=0;e<path.size();++e){
            const auto& tr=path[e];
            const auto actual_begin=tr.evaluate(0.f),actual_end=tr.evaluate(tr.T);
            if((actual_begin.values-tr.start.values).norm()>endpoint_tolerance_||
               (actual_end.values-tr.end.values).norm()>endpoint_tolerance_)
                return {false,0,0,0,0,"polynomial_boundary_mismatch"};
            if(e&&((path[e-1].end.values-tr.start.values).norm()>endpoint_tolerance_||
                   !terminalMetadataMatches(path[e-1],tr.metadata)))
                return {false,0,0,0,0,"flat_edge_stitching"};
            if(e&&(previous_flat.values-actual_begin.values).norm()>endpoint_tolerance_)
                return {false,0,0,0,0,"polynomial_junction_mismatch"};
            FloatSamples samples;
            if(!validate(tr,stats,&samples))return {false,0,0,0,0,"float_simd_discrete_rejection"};
            BoundaryDiagnostic d;d.kind=e?1:0;d.edge_index=e;d.time=double(elapsed);
            Eigen::VectorXd actual=samples.states.front().cast<double>();
            Eigen::VectorXd ref=e?previous_end:requested_start_.template cast<double>();
            d.physical=physicalBoundaryError(ref,actual);
            const auto& z=e?previous_flat:start_;
            d.flat_position_error=double((z.output()-actual_begin.output()).norm());
            d.flat_velocity_error=double((z.derivative(1)-actual_begin.derivative(1)).norm());
            previous_flat=actual_end;
            match&=d.physical.maximum()<=double(endpoint_tolerance_);stats.boundary_diagnostics.push_back(d);
            previous_end=samples.states.back().cast<double>();
            for(std::size_t k=e?1:0;k<samples.states.size();++k){
                Eigen::VectorXd state=samples.states[k].cast<double>();
                if constexpr(Quad3)if(!all.states.empty()&&all.states.back().segment<4>(3).dot(state.segment<4>(3))<0)state.segment<4>(3)*=-1;
                all.states.push_back(std::move(state));all.times.push_back(double(elapsed+samples.times[k]));
            }
            for(const auto& u:samples.controls)all.controls.emplace_back(u.cast<double>());
            elapsed+=tr.T;
        }
        BoundaryDiagnostic d;d.kind=2;d.edge_index=path.size()-1;d.time=double(elapsed);
        d.physical=physicalBoundaryError(previous_end,requested_goal_.template cast<double>());
        d.flat_position_error=double((previous_flat.output()-goal_.output()).norm());
        d.flat_velocity_error=double((previous_flat.derivative(1)-goal_.derivative(1)).norm());
        match&=d.physical.maximum()<=double(endpoint_tolerance_);
        stats.boundary_diagnostics.push_back(d);stats.physical_boundary_match=match;
        return {true,0,0,0,0,{}};
    }
private:
    FlatState lift(const PhysicalState& x,float speed)const{
        FlatState z;
        if constexpr(Uni){
            z.setDerivative(0,x.template head<2>());
            FlatOutput v;v<<speed*std::cos(x(2)),speed*std::sin(x(2));z.setDerivative(1,v);
        }else if constexpr(Quad3){
            Eigen::Quaternionf q(x(6),x(3),x(4),x(5));
            if(q.norm()<=batch_.epsilon)throw std::invalid_argument("Zero quaternion");
            q.normalize();Eigen::Matrix3f R=q.toRotationMatrix(),skew;
            Eigen::Vector3f w=x.template tail<3>();
            skew<<0.f,-w.z(),w.y(),w.z(),0.f,-w.x(),-w.y(),w.x(),0.f;
            Eigen::Matrix3f rd=R*skew;
            const float den=R(0,0)*R(0,0)+R(1,0)*R(1,0);
            if(den<=batch_.epsilon*batch_.epsilon)throw std::invalid_argument("Singular yaw lifting");
            FlatOutput y,v;y.template head<3>()=x.template head<3>();y(3)=std::atan2(R(1,0),R(0,0));
            v.template head<3>()=x.template segment<3>(7);
            v(3)=(R(0,0)*rd(1,0)-R(1,0)*rd(0,0))/den;
            z.setDerivative(0,y);z.setDerivative(1,v);
        }else{
            z.setDerivative(0,x.template head<2>());z.setDerivative(1,x.template segment<2>(3));
        }
        return z;
    }
    bool stateValid(const PhysicalState& x)const{
        PhysicalBatch p;for(int k=0;k<StateDim;++k)p.states[k][0]=x(k);
        stateBatch(batch_,1,*geometry_,p);return p.valid_mask==1;
    }
    bool validate(const FlatTrajectory& tr,PlanResult& stats,FloatSamples* samples)const{
        if constexpr(Uni){
            if(tr.metadata.branch!=initial_.branch){++stats.df_rejected;return false;}
            auto cert=certifyUnicycleSpeed(tr,batch_.epsilon);
            if(cert!=SpeedCertificate::Valid){
                ++stats.speed_certificate_rejected;++stats.df_rejected;
                if(cert==SpeedCertificate::Uncertain)++stats.speed_certificate_uncertain;
                ++stats.df_failures[std::size_t(FeasibilityFailure::SpeedZero)];return false;
            }
        }
        return validateFloatDiscrete(tr,batch_,branch_,*geometry_,max_step_,max_samples_,stats,samples);
    }
    const SphereGeometry* geometry_;
    PhysicalBatchConfig batch_;
    FlatState start_,goal_;FlatOutput lower_,upper_;Metadata initial_{};
    PhysicalState requested_start_,requested_goal_;
    int branch_{1};float max_step_{.005f},endpoint_tolerance_{.001f};std::size_t max_samples_{20000};
};
}
