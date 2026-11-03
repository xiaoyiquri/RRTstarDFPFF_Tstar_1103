#include "pff/r2_float_kernel.h"
#include <cmath>
#include <stdexcept>
namespace rrtpff {
#ifdef RRTPFF_HAS_AVX2
void distanceBatchFloatAVX2(const float*,const float*,const float*,const float*,int,std::size_t,std::size_t,float*);
#endif
void distanceBatchFloat(const float* x,const float* q,const float* w,const float* p,int dim,
    std::size_t count,std::size_t stride,float* out,SIMDBackend backend) {
    std::size_t start=0;
#ifdef RRTPFF_HAS_AVX2
    if(backend==SIMDBackend::AVX2&&simdRuntimeAVX2()) {
        distanceBatchFloatAVX2(x,q,w,p,dim,count,stride,out);start=count/8*8;
    }
#endif
    for(std::size_t i=start;i<count;++i) {
        float sum=0.f;
        for(int k=0;k<dim;++k) {
            float d=x[k*stride+i]-q[k];if(p[k]>0.f)d=std::remainder(d,p[k]);
            sum+=d*d*w[k];
        }
        out[i]=sum;
    }
}
namespace production {
#ifdef RRTPFF_HAS_AVX2
void r2EnergyBatchFloatAVX2(const float*,const float*,const float*,const float*,const unsigned char*,
                           const float*,int,std::size_t,float*,float*,float*);
#endif
void r2EnergyBatchFloat(const float* y0,const float* v0,const float* yf,const float* vf,
    const unsigned char* fixed,const float* R,int dim,std::size_t count,float* a,float* b,float* c) {
#ifdef RRTPFF_HAS_AVX2
    if(simdRuntimeAVX2()) {
        r2EnergyBatchFloatAVX2(y0,v0,yf,vf,fixed,R,dim,count,a,b,c);return;
    }
#endif
    throw std::runtime_error("Production r2 batch requires AVX2");
}
}
}

