#include "simd/simd_backend.h"
#include <immintrin.h>
namespace rrtpff {
void distanceBatchFloatAVX2(const float* x,const float* q,const float* w,const float* periods,
    int dim,std::size_t count,std::size_t stride,float* out) {
    std::size_t i=0;
    for(;i+8<=count;i+=8) {
        __m256 sum=_mm256_setzero_ps();
        for(int k=0;k<dim;++k) {
            __m256 d=_mm256_sub_ps(_mm256_loadu_ps(x+k*stride+i),_mm256_set1_ps(q[k]));
            if(periods[k]>0.f) {
                const auto p=_mm256_set1_ps(periods[k]);
                auto n=_mm256_round_ps(_mm256_div_ps(d,p),_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);
                d=_mm256_sub_ps(d,_mm256_mul_ps(n,p));
            }
            sum=_mm256_add_ps(sum,_mm256_mul_ps(_mm256_mul_ps(d,d),_mm256_set1_ps(w[k])));
        }
        _mm256_storeu_ps(out+i,sum);
    }
    // Baseline tail is dispatched by the wrapper.
}
namespace production {
void r2EnergyBatchFloatAVX2(const float* y0,const float* v0,const float* yf,const float* vf,
    const unsigned char* fixed,const float* R,int dim,std::size_t,float* a,float* b,float* c) {
    __m256 d[4],v[4],f[4];
    for(int k=0;k<dim;++k) {
        d[k]=_mm256_sub_ps(_mm256_loadu_ps(yf+8*k),_mm256_loadu_ps(y0+8*k));
        v[k]=_mm256_loadu_ps(v0+8*k);f[k]=_mm256_loadu_ps(vf+8*k);
    }
    auto dot=[&](const __m256* x,const __m256* y) {
        __m256 out=_mm256_setzero_ps();
        for(int k=0;k<dim;++k)for(int j=0;j<dim;++j)
            out=_mm256_add_ps(out,_mm256_mul_ps(_mm256_mul_ps(x[k],_mm256_set1_ps(R[k*dim+j])),y[j]));
        return out;
    };
    __m256 vv=dot(v,v),ff=dot(f,f),vff=dot(v,f),dd=dot(d,d),vd=dot(v,d),s[4];
    for(int k=0;k<dim;++k)s[k]=_mm256_add_ps(v[k],f[k]);
    const auto sd=dot(s,d);
    __m256 af=_mm256_mul_ps(_mm256_set1_ps(12.f),dd),bf=_mm256_mul_ps(_mm256_set1_ps(-12.f),sd);
    __m256 cf=_mm256_mul_ps(_mm256_set1_ps(4.f),_mm256_add_ps(_mm256_add_ps(vv,vff),ff));
    __m256 au=_mm256_mul_ps(_mm256_set1_ps(3.f),dd),bu=_mm256_mul_ps(_mm256_set1_ps(-6.f),vd);
    __m256 cu=_mm256_mul_ps(_mm256_set1_ps(3.f),vv);
    float mask[8];for(int i=0;i<8;++i)mask[i]=fixed[i]?1.f:0.f;
    auto m=_mm256_cmp_ps(_mm256_loadu_ps(mask),_mm256_setzero_ps(),_CMP_GT_OQ);
    _mm256_storeu_ps(a,_mm256_blendv_ps(au,af,m));
    _mm256_storeu_ps(b,_mm256_blendv_ps(bu,bf,m));
    _mm256_storeu_ps(c,_mm256_blendv_ps(cu,cf,m));
}
}
}

