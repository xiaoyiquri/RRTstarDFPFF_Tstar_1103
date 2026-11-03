#pragma once
#include <algorithm>
#include <cmath>
#include <vector>

namespace rrtpff {
inline double factorial(int n) { double f = 1;
 for (int i=2; i<=n; ++i) f *= i;
 return f;
 }
inline double binomial(int n, int k) { return factorial(n)/(factorial(k)*factorial(n-k));
 }

// Ascending coefficients. Recursive derivative isolation partitions the interval
// into monotone pieces and also catches even-multiplicity roots at critical points.
// This differs from sign-change-only uniform scans in the old time solver.
inline long double polynomialValue(const std::vector<long double>& c, long double x) {
    long double v=0;
 for (auto it=c.rbegin(); it!=c.rend(); ++it) v=v*x+*it;
 return v;
}
inline long double polynomialScale(const std::vector<long double>& c, long double x) {
    long double v=0;
 for (auto it=c.rbegin(); it!=c.rend(); ++it) v=v*std::abs(x)+std::abs(*it);
 return v;
}
inline std::vector<double> realRootsInInterval(std::vector<long double> c, double lo, double hi) {
    while(c.size()>1 && c.back()==0) c.pop_back();
    std::vector<double> roots;
    if(c.size()<2 || !(hi>=lo)) return roots;
    if(c.size()==2) {
        const double x=static_cast<double>(-c[0]/c[1]);
        if(std::isfinite(x) && x>=lo && x<=hi) roots.push_back(x);
        return roots;
    }
    std::vector<long double> derivative(c.size()-1);
    for(std::size_t k=1;k<c.size();++k) derivative[k-1]=k*c[k];
    auto cuts=realRootsInInterval(derivative,lo,hi);
    cuts.insert(cuts.begin(),lo);
 cuts.push_back(hi);
    auto nearZero=[&](double t) {
        return std::abs(polynomialValue(c,t)) <= 1e-13L * std::max(1e-300L,polynomialScale(c,t));
    };
    for(double t:cuts) if(nearZero(t)) roots.push_back(t);
    for(std::size_t i=1;i<cuts.size();++i) {
        double a=cuts[i-1], b=cuts[i];
        long double fa=polynomialValue(c,a), fb=polynomialValue(c,b);
        if(fa==0 || fb==0 || (fa<0)==(fb<0)) continue;
        for(int j=0;j<100;++j) {
            double m=a+(b-a)*.5;
            long double fm=polynomialValue(c,m);
            if(fm==0) {a=b=m;
 break;
}
            if((fa<0)!=(fm<0)) b=m;
            else {a=m;
fa=fm;
}
            if(b-a<1e-12*std::max(1.0,std::abs(m))) break;
        }
        roots.push_back(a+(b-a)*.5);
    }
    std::sort(roots.begin(),roots.end());
    roots.erase(std::unique(roots.begin(),roots.end(),[](double a,double b){
        return std::abs(a-b)<=1e-10*std::max({1.0,std::abs(a),std::abs(b)});
    }),roots.end());
    return roots;
}
}
