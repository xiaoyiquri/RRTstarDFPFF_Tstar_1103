#pragma once
#include "common/flat_trajectory.h"
#include "simd/simd_backend.h"
#include <Eigen/Cholesky>
#include <array>
#include <vector>
#include <algorithm>
#include <limits>
#include <cmath>
namespace rrtpff::production {
struct FloatPFFConfig {
    float min_time{.001f}, max_time{5.f}, fixed_time{1.f};
    bool free_terminal_time{true}, defer_trajectories{true};
    std::size_t additional_time_samples{};
};
// Degree <= 4 root isolation: derivative roots split monotone intervals.
// No Ferrari-only formula, dynamic matrices, long double or double fallback.
inline float horner(const std::array<float,5>& p,int degree,float x) {
    float y=p[degree]; for(int k=degree-1;k>=0;--k) y=y*x+p[k]; return y;
}
inline std::vector<float> floatRoots(std::array<float,5> p,int degree,float lo,float hi) {
    std::vector<float> roots;
    float scale=0.f; for(int k=0;k<=degree;++k) scale=std::max(scale,std::abs(p[k]));
    if (!(scale>0.f) || !std::isfinite(scale) || !(lo<=hi)) return roots;
    for(int k=0;k<=degree;++k) p[k]/=scale;
    while(degree>0 && p[degree]==0.f) --degree;
    if(!degree) return roots;
    if(degree==1) {
        float x=-p[0]/p[1]; if(std::isfinite(x)&&x>=lo&&x<=hi) roots.push_back(x);
        return roots;
    }
    std::array<float,5> dp{}; for(int k=1;k<=degree;++k) dp[k-1]=float(k)*p[k];
    auto split=floatRoots(dp,degree-1,lo,hi);
    split.insert(split.begin(),lo); split.push_back(hi);
    auto add=[&](float x) {
        if(roots.empty() || std::abs(x-roots.back())>16.f*std::numeric_limits<float>::epsilon()*(1.f+std::abs(x)))
            roots.push_back(x);
    };
    auto nearlyZero=[&](float x,float value) {
        float bound=0.f,power=1.f;
        for(int k=0;k<=degree;++k) { bound+=std::abs(p[k])*power; power*=std::abs(x); }
        return std::abs(value)<=16.f*std::numeric_limits<float>::epsilon()*bound;
    };
    for(std::size_t k=0;k<split.size();++k) {
        const float x=split[k], fx=horner(p,degree,x);
        if(nearlyZero(x,fx)) add(x); // Includes even-multiplicity roots.
        if(k+1==split.size()) break;
        float a=x,b=split[k+1],fa=fx,fb=horner(p,degree,b);
        if(!std::isfinite(fa)||!std::isfinite(fb)||fa==0.f||fb==0.f||
           std::signbit(fa)==std::signbit(fb)) continue;
        for(int it=0;it<64;++it) {
            const float mid=a+(b-a)*.5f;
            if(mid==a||mid==b) break;
            const float fm=horner(p,degree,mid);
            if(fm==0.f) { a=b=mid; break; }
            if(std::signbit(fm)==std::signbit(fa)) {a=mid;fa=fm;} else b=mid;
        }
        add(a+(b-a)*.5f);
    }
    std::sort(roots.begin(),roots.end());
    roots.erase(std::unique(roots.begin(),roots.end()),roots.end());
    return roots;
}
template<int Dim> struct FloatConstraint {
    using State=FlatState<Dim,2,float>;
    State target{};
    bool fixed_velocity{false};
    static FloatConstraint output(const typename State::Output& y) {
        FloatConstraint c;c.target.setDerivative(0,y);return c;
    }
    static FloatConstraint full(const State& x) {
        FloatConstraint c;c.target=x;c.fixed_velocity=true;return c;
    }
};
template<int Dim,class Metadata=std::monostate> struct FloatPFFResult {
    using State=FlatState<Dim,2,float>;
    using Trajectory=FlatTrajectory<Dim,2,Metadata,float>;
    bool success{},deferred{};
    FloatConstraint<Dim> boundary{};
    float T{},cost{std::numeric_limits<float>::infinity()};
    State terminal_state{};
    Trajectory trajectory{};
};
void r2EnergyBatchFloat(const float*,const float*,const float*,const float*,const unsigned char*,
                       const float*,int,std::size_t,float*,float*,float*);
template<int Dim,class Metadata=std::monostate> class R2FloatSolver {
public:
    using State=FlatState<Dim,2,float>;
    using Output=typename State::Output;
    using Constraint=FloatConstraint<Dim>;
    using Result=FloatPFFResult<Dim,Metadata>;
    using Weight=Eigen::Matrix<float,Dim,Dim>;
    using Energy=std::array<float,3>; // A/T^3 + B/T^2 + C/T
    explicit R2FloatSolver(Weight R=Weight::Identity(),FloatPFFConfig config={}):R_(R),config_(config) {
        if(!R.allFinite()||!R.isApprox(R.transpose(),1e-5f)||Eigen::LLT<Weight>(R).info()!=Eigen::Success ||
           !(config.min_time>0.f)||!std::isfinite(config.max_time)||config.max_time<config.min_time ||
           !(config.fixed_time>0.f)||!std::isfinite(config.fixed_time)||config.additional_time_samples>128)
            throw std::invalid_argument("Invalid r2 float PFF parameters");
    }
    const Weight& weight()const{return R_;}
    const FloatPFFConfig& config()const{return config_;}
    float dot(const Output& a,const Output& b)const {
        float sum=0.f;
        for(int i=0;i<Dim;++i) for(int j=0;j<Dim;++j) sum+=a(i)*R_(i,j)*b(j);
        return sum;
    }
    Energy energy(const State& x,const Constraint& c)const {
        const Output d=c.target.output()-x.output(),v=x.derivative(1),vf=c.target.derivative(1);
        if(c.fixed_velocity)
            return {12.f*dot(d,d),-12.f*dot(v+vf,d),4.f*(dot(v,v)+dot(v,vf)+dot(vf,vf))};
        return {3.f*dot(d,d),-6.f*dot(v,d),3.f*dot(v,v)};
    }
    Result fixedTime(const State& x,const Constraint& c,float T)const {
        Result out;out.T=T;out.boundary=c;
        if(!x.finite()||!c.target.finite()||!(T>0.f)||!std::isfinite(T)) return out;
        const Output v=x.derivative(1),d=c.target.output()-x.output()-T*v;
        const Output e=c.target.derivative(1)-v;
        auto& tr=out.trajectory;tr.T=T;tr.normalized=true;tr.start=x;
        tr.coefficients.col(0)=x.output();tr.coefficients.col(1)=T*v;
        if(c.fixed_velocity) {
            tr.coefficients.col(2)=3.f*d-T*e;
            tr.coefficients.col(3)=-2.f*d+T*e;
            const Output residual=d-.5f*T*e;
            out.cost=T+12.f*dot(residual,residual)/(T*T*T)+dot(e,e)/T;
        } else {
            tr.coefficients.col(2)=1.5f*d;tr.coefficients.col(3)=-.5f*d;
            out.cost=T+3.f*dot(d,d)/(T*T*T);
        }
        tr.end=c.fixed_velocity?c.target:tr.evaluate(T);
        tr.end.setDerivative(0,c.target.output());
        tr.cost=out.cost;out.terminal_state=tr.end;
        // A physical-time cubic needs jerk even though state/control order is 2.
        out.success=std::isfinite(out.cost)&&out.cost>=T&&tr.coefficients.allFinite();
        return out;
    }
    bool materialize(const State& x,Result& out)const {
        if(!out.success) return false;
        if(!out.deferred) return true;
        out=fixedTime(x,out.boundary,out.T);return out.success;
    }
    
    std::vector<Result> candidatesWithEnergy(
    const State& x,
    const Constraint& c,
    const Energy& e) const
{
    std::vector<Result> out;
    float selected_time = config_.fixed_time;

    if (config_.free_terminal_time) {
        // J(T) = T + A/T^3 + B/T^2 + C/T
        // J'(T)=0:
        // T^4 - C*T^2 - 2*B*T - 3*A = 0
        const float A = e[0];
        const float B = e[1];
        const float C = e[2];

        if (!std::isfinite(A) ||
            !std::isfinite(B) ||
            !std::isfinite(C)) {
            return out;
        }

        // Root bound, NOT a user-selected maximum trajectory time.
        //
        // For T > scale, each of
        // |C|T^2, 2|B|T, 3|A|
        // is smaller than T^4/3, so no positive root exists there.
        const float scale = std::max({
            1.f,
            std::sqrt(std::abs(C)) * std::sqrt(3.f),
            std::cbrt(std::abs(B)) * std::cbrt(6.f),
            std::sqrt(std::sqrt(std::abs(A))) * std::sqrt(3.f)
        });

        if (!std::isfinite(scale)) {
            return out;
        }

        // u = T / scale.
        // Ascending polynomial coefficients for floatRoots().
        // Divide successively to avoid forming scale^4.
        std::array<float, 5> polynomial{{
            -3.f * ((((A / scale) / scale) / scale) / scale),
            -2.f * (((B / scale) / scale) / scale),
            -((C / scale) / scale),
            0.f,
            1.f
        }};

        const auto roots = floatRoots(polynomial, 4, 0.f, 1.f);

        selected_time = std::numeric_limits<float>::infinity();

        for (float u : roots) {
            const float T = u * scale;

            if (std::isfinite(T) && T > 0.f &&
                T < selected_time) {
                selected_time = T;
            }
        }

        // No positive root: reject this endpoint connection.
        if (!std::isfinite(selected_time)) {
            return out;
        }
    }

    // Evaluate ONLY the selected time.
    // If construction fails, do not try another root or a boundary.
    auto result = fixedTime(x, c, selected_time);

    if (!result.success) {
        return out;
    }

    if (config_.defer_trajectories) {
        result.deferred = true;
        result.trajectory = {};
    }

    out.push_back(std::move(result));
    return out;
}
    
    /*
    std::vector<Result> candidatesWithEnergy(const State& x,const Constraint& c,const Energy& e)const {
        std::vector<float> times;
        if(config_.free_terminal_time) {
            // Use u=T/max_time to bound the isolation interval in [0,1].
            const float m=config_.max_time;
            std::array<float,5> p{{-3.f*e[0],-2.f*e[1]*m,-e[2]*m*m,0.f,m*m*m*m}};
            for(float u:floatRoots(p,4,config_.min_time/m,1.f)) times.push_back(u*m);
            times.push_back(config_.min_time);times.push_back(config_.max_time);
        } else times.push_back(config_.fixed_time);
        for(std::size_t k=1;k<=config_.additional_time_samples;++k)
            times.push_back(config_.min_time+(config_.max_time-config_.min_time)*
                            float(k)/float(config_.additional_time_samples+1));
        std::sort(times.begin(),times.end());
        times.erase(std::unique(times.begin(),times.end(),[](float a,float b) {
            return std::abs(a-b)<=8.f*std::numeric_limits<float>::epsilon()*(1.f+std::abs(a));
        }),times.end());
        std::vector<Result> out;
        for(float T:times) {
            auto result=fixedTime(x,c,T); // Cost is evaluated using the real boundary expressions.
            if(result.success) {
                if(config_.defer_trajectories) {result.deferred=true;result.trajectory={};}
                out.push_back(std::move(result));
            }
        }
        std::stable_sort(out.begin(),out.end(),[](const auto& a,const auto& b) {
            return a.cost==b.cost?a.T<b.T:a.cost<b.cost;
        });
        return out;
    }
    */
    std::vector<Result> candidates(const State& x,const Constraint& c)const {
        return candidatesWithEnergy(x,c,energy(x,c));
    }
    Result solve(const State& x,const Constraint& c)const {
        auto list=candidates(x,c);if(list.empty())return {};
        auto out=list.front();materialize(x,out);return out;
    }
private:
    Weight R_;FloatPFFConfig config_;
};
template<int Dim,class Metadata=std::monostate> class R2FloatBatch {
public:
    using Solver=R2FloatSolver<Dim,Metadata>;using State=typename Solver::State;
    using Constraint=typename Solver::Constraint;using Result=typename Solver::Result;
    explicit R2FloatBatch(Solver s={}):solver_(std::move(s)){}
    bool materialize(const State& x,Result& r)const{return solver_.materialize(x,r);}
    bool deferred()const{return solver_.config().defer_trajectories;}
    std::array<std::vector<Result>,8> candidatesBatch(const std::array<State,8>& x,
        const std::array<Constraint,8>& c,std::size_t count,bool enabled,std::size_t* lanes)const {
        if(count>8)throw std::invalid_argument("r2 float batch overflow");
        std::array<std::vector<Result>,8> out;
        if(!count)return out;
        if(enabled&&simdRuntimeAVX2()) {
            float y0[Dim*8]{},v0[Dim*8]{},yf[Dim*8]{},vf[Dim*8]{},R[Dim*Dim]{},a[8],b[8],cc[8];
            unsigned char fixed[8]{};
            for(int k=0;k<Dim;++k) {
                for(int j=0;j<Dim;++j)R[k*Dim+j]=solver_.weight()(k,j);
                for(std::size_t i=0;i<count;++i) {
                    y0[k*8+i]=x[i].output()(k);v0[k*8+i]=x[i].derivative(1)(k);
                    yf[k*8+i]=c[i].target.output()(k);vf[k*8+i]=c[i].target.derivative(1)(k);
                }
            }
            for(std::size_t i=0;i<count;++i)fixed[i]=c[i].fixed_velocity;
            r2EnergyBatchFloat(y0,v0,yf,vf,fixed,R,Dim,count,a,b,cc);
            if(lanes)*lanes+=count;
            for(std::size_t i=0;i<count;++i)out[i]=solver_.candidatesWithEnergy(x[i],c[i],{a[i],b[i],cc[i]});
        } else for(std::size_t i=0;i<count;++i)out[i]=solver_.candidates(x[i],c[i]);
        return out;
    }
private:Solver solver_;
};
} // namespace rrtpff::production
