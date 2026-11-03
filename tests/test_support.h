#pragma once
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
inline void require(bool condition,const std::string& message) {
    if(!condition) throw std::runtime_error(message);
}
inline void close(double actual,double expected,double tolerance,const std::string& message) {
    require(std::isfinite(actual)&&std::isfinite(expected)&&std::abs(actual-expected)<=tolerance*(1+std::abs(expected)),message);
}
template<class Function> int runTest(Function function) {
    try {function();
std::cout<<"PASS\n";
return 0;
}
    catch(const std::exception& e) {
        std::cerr<<"FAIL: "<<e.what()<<'\n';
return 1;
}
}
