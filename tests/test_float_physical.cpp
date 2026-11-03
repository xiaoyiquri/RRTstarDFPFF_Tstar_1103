#include "tests/test_support.h"
#include "production/float_validator.h"
#include "simd/simd_backend.h"
#include "systems/batch_config.h"
#include "common/boundary_diagnostics.h"
using namespace rrtpff;
template<class DF> void parity(typename DF::Config c,int branch){
    constexpr int D=DF::FlatState::OutputDim;
    FlatTrajectory<D,2,std::monostate,float> tr;tr.T=2;tr.normalized=true;
    tr.coefficients(0,1)=.6f;tr.coefficients(0,2)=.05f;tr.coefficients(0,3)=.01f;
    tr.coefficients(1,0)=1;tr.coefficients(1,2)=.01f;
    if constexpr(D==4){tr.coefficients(3,1)=.04f;tr.coefficients(3,2)=.01f;}
    tr.start=tr.evaluate(0);tr.end=tr.evaluate(tr.T);
    DynoProblemData data;constexpr int NX=DF::PhysicalState::RowsAtCompileTime,NU=DF::PhysicalControl::RowsAtCompileTime;
    data.lower_state=Eigen::VectorXd::Constant(NX,-100);data.upper_state=-data.lower_state;
    data.lower_control=Eigen::VectorXd::Constant(NU,-100);data.upper_control=-data.lower_control;
    auto config=makeBatchConfig<DF>(c,data);
    SphereGeometry geometry;geometry.dimension=D==4?3:2;geometry.robot_radius=.1f;geometry.build();
    BernsteinCurve curve;require(production::floatBernstein(tr,curve),"float normalized curve conversion");
    require(curve.coefficients[4][0][0]==0,"snap in SIMD curve is zero");
    require(curve.coefficients[3][0][0]!=0,"nonzero jerk retained");
    float times[8]{0,.01f,.15f,.3f,.5f,.7f,.91f,1};PhysicalBatch out;
    curveBatch(curve,times,config,branch,255,geometry,out);
    require(out.valid_mask==255,"all float physical samples feasible");
    for(int lane=0;lane<8;++lane){
        typename DF::FlatState f;typename DF::FlatControl uflat;
        for(int d=0;d<DF::FlatState::IntegratorOrder;++d)
            f.setDerivative(d,tr.derivative(times[lane]*tr.T,d).template cast<double>());
        uflat.values=tr.derivative(times[lane]*tr.T,DF::FlatState::IntegratorOrder).template cast<double>();
        typename DF::FlatTrajectory::MetadataType metadata{};
        if constexpr(std::is_same_v<DF,UnicycleDF>)
            require(UnicycleDF::makeMetadata(branch*.3,metadata,c),"signed metadata");
        typename DF::PhysicalState state;typename DF::PhysicalControl u;
        require(DF::sampleFeasible(f,uflat,metadata,state,u,c),"double reference feasible");
        Eigen::VectorXd actual(NX);for(int k=0;k<NX;++k)actual(k)=out.states[k][lane];
        require(physicalBoundaryError(state,actual).maximum()<1e-4,"float reconstructed state vs double reference");
        for(int k=0;k<NU;++k)close(out.controls[k][lane],u(k),1e-4,"float reconstructed control vs double reference");
    }
    PlanResult stats;production::FloatSamples samples;
    require(production::validateFloatDiscrete(tr,config,branch,geometry,.005f,20000,stats,&samples),"float discrete validator and capture");
    require(samples.states.size()==samples.controls.size()+1,"one control per interval");
    for(std::size_t k=1;k<samples.times.size();++k)
        require(samples.times[k]-samples.times[k-1]<=.005f,"actual physical time gaps below max_step");
}
int main(){return runTest([]{
    require(simdRuntimeAVX2(),"AVX2 needed for production validation tests");
    parity<Quad2DDF>(Quad2DDF::Config{},1);
    auto q3=Quad3DDF::Config{};auto map=std::make_shared<WrenchControlMap>();
    map->lower.setConstant(-100);map->upper.setConstant(100);q3.control_map=map;
    parity<Quad3DDF>(q3,1);
    auto u=UnicycleDF::Config{};u.allow_reverse=true;u.velocity_lower=-1;u.velocity_upper=1;
    parity<UnicycleDF>(u,1);parity<UnicycleDF>(u,-1);
});}
