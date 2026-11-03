#pragma once
#include "pff/r2_float_kernel.h"
namespace rrtpff::production {
struct FloatInterval {float lo{},hi{};};
inline float down(float x){return std::nextafter(x,-std::numeric_limits<float>::infinity());}
inline float up(float x){return std::nextafter(x,std::numeric_limits<float>::infinity());}
inline FloatInterval add(FloatInterval a,FloatInterval b){return {down(a.lo+b.lo),up(a.hi+b.hi)};}
inline FloatInterval mul(FloatInterval a,FloatInterval b){
    const float p[4]={a.lo*b.lo,a.lo*b.hi,a.hi*b.lo,a.hi*b.hi};
    return {down(*std::min_element(p,p+4)),up(*std::max_element(p,p+4))};
}
inline FloatInterval divPositive(FloatInterval a,float x){return {down(a.lo/x),up(a.hi/x)};}
enum class SpeedCertificate {Valid,ZeroOrBelowThreshold,Uncertain};
template<class Metadata> SpeedCertificate certifyUnicycleSpeed(
    const FlatTrajectory<2,2,Metadata,float>& tr,float epsilon,int max_depth=20,int max_cells=4096) {
    if(!(tr.T>0.f)||!tr.coefficients.allFinite()||!tr.normalized)
        return SpeedCertificate::Uncertain;
    std::array<float,5> q{};
    std::array<FloatInterval,5> bounded{};
    for(int axis=0;axis<2;++axis) {
        float v[3];FloatInterval vi[3];
        for(int k=0;k<3;++k){
            FloatInterval p{tr.coefficients(axis,k+1),tr.coefficients(axis,k+1)};
            vi[k]=divPositive(mul(p,{float(k+1),float(k+1)}),tr.T);
            v[k]=float(k+1)*tr.coefficients(axis,k+1)/tr.T;
        }
        for(int k=0;k<3;++k)for(int j=0;j<3;++j) {
            q[k+j]+=v[k]*v[j];bounded[k+j]=add(bounded[k+j],mul(vi[k],vi[j]));
        }
    }
    std::array<float,5> dq{};for(int k=1;k<5;++k)dq[k-1]=float(k)*q[k];
    auto stationary=floatRoots(dq,3,0.f,1.f);stationary.push_back(0.f);stationary.push_back(1.f);
    // Only early rejection uses approximate stationary roots. Acceptance always
    // requires outward interval bounds, so a missed/near-multiple root cannot certify a stop.
    for(float s:stationary)if(tr.velocity(s*tr.T).squaredNorm()<=epsilon*epsilon)
        return SpeedCertificate::ZeroOrBelowThreshold;
    std::array<FloatInterval,5> bernstein{};
    const int choose4[5]={1,4,6,4,1};
    const int choose[5][5]={{1,0,0,0,0},{1,1,0,0,0},{1,2,1,0,0},{1,3,3,1,0},{1,4,6,4,1}};
    for(int l=0;l<5;++l)for(int k=0;k<=l;++k)
        bernstein[l]=add(bernstein[l],divPositive(mul(bounded[k],{float(choose[l][k]),float(choose[l][k])}),float(choose4[k])));
    struct Cell{std::array<FloatInterval,5>b;int depth;};
    std::vector<Cell> stack{{bernstein,0}};int visited=0;
    const float threshold=up(epsilon*epsilon);
    while(!stack.empty()){
        Cell cell=stack.back();stack.pop_back();
        if(++visited>max_cells)return SpeedCertificate::Uncertain;
        float lo=std::numeric_limits<float>::infinity(),hi=-lo;
        for(auto b:cell.b){lo=std::min(lo,b.lo);hi=std::max(hi,b.hi);}
        if(!std::isfinite(lo)||!std::isfinite(hi))return SpeedCertificate::Uncertain;
        if(lo>threshold)continue;
        if(hi<=threshold)return SpeedCertificate::ZeroOrBelowThreshold;
        if(cell.depth>=max_depth)return SpeedCertificate::Uncertain;
        auto work=cell.b;std::array<FloatInterval,5>left{},right{};
        left[0]=work[0];right[4]=work[4];
        for(int n=4;n>0;--n){
            for(int k=0;k<n;++k)work[k]=mul(add(work[k],work[k+1]),{.5f,.5f});
            left[5-n]=work[0];right[n-1]=work[n-1];
        }
        stack.push_back({right,cell.depth+1});stack.push_back({left,cell.depth+1});
    }
    return SpeedCertificate::Valid;
}
}

