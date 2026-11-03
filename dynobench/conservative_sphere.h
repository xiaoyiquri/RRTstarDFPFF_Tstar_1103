#pragma once
#include "simd/physical_batch.h"
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
namespace rrtpff {
// Cold input conversion. Enclose the reference double sphere, including the
// center's quantization error. Every error-bound operation rounds outward.
inline Sphere conservativeSphere(const std::array<double,3>& center,double radius,int dimension){
    Sphere s;double error2=0;const double inf=std::numeric_limits<double>::infinity();
    for(int k=0;k<dimension;++k){
        s.center[k]=float(center[k]);
        if(!std::isfinite(s.center[k]))throw std::invalid_argument("Sphere center exceeds float range");
        const double error=std::nextafter(std::abs(center[k]-double(s.center[k])),inf);
        error2=std::nextafter(error2+std::nextafter(error*error,inf),inf);
    }
    const double upper=std::nextafter(radius+std::nextafter(std::sqrt(error2),inf),inf);
    s.radius=std::nextafter(float(upper),std::numeric_limits<float>::infinity());
    if(!(s.radius>0)||!std::isfinite(s.radius))throw std::invalid_argument("Invalid float sphere radius");
    return s;
}
}
