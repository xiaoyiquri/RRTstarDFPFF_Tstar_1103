#include "dynobench/uni_adapter.h"
#include "dynobench/quad2d_adapter.h"
#include "dynobench/quad3d_adapter.h"
#include "tests/test_support.h"
using namespace rrtpff;

int main() {
    return runTest([] {

        UniAdapter u;UniAdapter::State x=UniAdapter::State::Zero();UniAdapter::Control c;c<<.1,.1;UniAdapter::State n;require(u.step(x,c,.01,n),"uni adapter step");
        Quad2DAdapter q2;Quad2DAdapter::State s2=Quad2DAdapter::State::Zero();Quad2DAdapter::Control c2;c2<<1,1;Quad2DAdapter::State n2;require(q2.step(s2,c2,.01,n2),"quad2d adapter step");
        Quad3DAdapter q3;Quad3DAdapter::State s3=Quad3DAdapter::State::Zero();s3(6)=1;Quad3DAdapter::Control c3;c3<<1,1,1,1;Quad3DAdapter::State n3;require(q3.step(s3,c3,.01,n3),"quad3d adapter step");
    }
    );
}
