#pragma once
#include "common/types.h"
#include "common/timer.h"
#include "common/flat_trajectory.h"
#include "simd/physical_batch.h"
namespace rrtpff::production {
struct FloatSamples {
    std::vector<Eigen::VectorXf> states,controls;
    std::vector<float> times;
};
inline float chooseFloat(int n,int k){
    float v=1.f;for(int i=1;i<=k;++i)v*=float(n-k+i)/float(i);return v;
}
template<int Dim,class Metadata> bool floatBernstein(
    const FlatTrajectory<Dim,2,Metadata,float>& tr,BernsteinCurve& curve) {
    if(!tr.normalized||!(tr.T>0.f)||!std::isfinite(tr.T)||!tr.coefficients.allFinite())return false;
    curve={};curve.dim=Dim;curve.order=2;curve.derivative_order=4;
    float time_scale=1.f;
    for(int d=0;d<=3;++d) {
        const int degree=3-d;
        for(int axis=0;axis<Dim;++axis)for(int l=0;l<=degree;++l) {
            float value=0.f;
            for(int k=0;k<=l;++k) {
                float factor=1.f;for(int j=1;j<=d;++j)factor*=float(k+j);
                value+=tr.coefficients(axis,k+d)*factor*time_scale*chooseFloat(l,k)/chooseFloat(degree,k);
            }
            if(!std::isfinite(value))return false;
            curve.coefficients[d][axis][l]=value;
        }
        time_scale/=tr.T;
    }
    return true; // Initialized snap remains exact zero.
}
template<int Dim,class Metadata> bool validateFloatDiscrete(
    const FlatTrajectory<Dim,2,Metadata,float>& tr,const PhysicalBatchConfig& config,int branch,
    const SphereGeometry& geometry,float step,std::size_t maximum,PlanResult& stats,FloatSamples* samples=nullptr) {
    Timer timer;++stats.simd_validation_queries;stats.validation_backend="simd_discrete_float8";
    BernsteinCurve curve;
    if(!floatBernstein(tr,curve)||!(step>0.f)) {++stats.df_rejected;return false;}
    // One extra interval leaves room for rounding of normalized sample times.
    const float n=std::ceil(tr.T/step)+1.f;
    if(!std::isfinite(n)||n<1.f||n+1.f>float(maximum)) {
        ++stats.df_rejected;++stats.df_failures[std::size_t(FeasibilityFailure::SampleLimit)];return false;
    }
    const std::size_t intervals=std::size_t(n),count=intervals+1,batches=(count+7)/8;
    if(samples){samples->states.resize(count);samples->controls.resize(intervals);samples->times.resize(count);}
    for(std::size_t b=0;b<batches;++b) {
        float times[8]{};std::size_t indices[8]{};unsigned active=0;
        for(int lane=0;lane<8;++lane){
            indices[lane]=b+std::size_t(lane)*batches;
            if(indices[lane]<count){active|=1u<<lane;times[lane]=float(indices[lane])/float(intervals);}
        }
        PhysicalBatch out;curveBatch(curve,times,config,branch,active,geometry,out);
        ++stats.simd_validation_batches;
        for(int lane=0;lane<8;++lane)if(active&(1u<<lane)){++stats.simd_flat_samples;++stats.simd_collision_samples;}
        stats.collision_check_count+=out.sphere_tests;
        if(out.valid_mask!=active){
            ++stats.rejected_candidates;
            if(out.df_mask!=active)++stats.df_rejected;
            else if(out.state_mask!=active)++stats.state_bound_rejected;
            else if(out.control_mask!=active)++stats.control_bound_rejected;
            else ++stats.collision_rejected;
            stats.simd_time+=timer.seconds();return false;
        }
        if(samples)for(int lane=0;lane<8;++lane)if(active&(1u<<lane)) {
            const auto i=indices[lane];samples->states[i].resize(config.state_dim);
            for(int k=0;k<config.state_dim;++k)samples->states[i](k)=out.states[k][lane];
            samples->times[i]=times[lane]*tr.T;
            if(i<intervals){samples->controls[i].resize(config.control_dim);
                for(int k=0;k<config.control_dim;++k)samples->controls[i](k)=out.controls[k][lane];}
        }
    }
    stats.simd_time+=timer.seconds();return true;
}
}

