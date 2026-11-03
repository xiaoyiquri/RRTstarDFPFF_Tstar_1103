#pragma once
#include <chrono>
namespace rrtpff {
class Timer {
    using Clock=std::chrono::steady_clock;
    Clock::time_point start_=Clock::now();
public:
    double seconds() const {return std::chrono::duration<double>(Clock::now()-start_).count();
}
};
}
