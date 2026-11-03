#include "benchmark/benchmark_runner.h"
#include <exception>
#include <iostream>
#include <string>
int main(int argc,char** argv) {
    if(argc!=3||std::string(argv[1])!="--config") {
        std::cerr<<"Usage: benchmark --config benchmark/config/linear_reference.yaml\n";
        return 1;
    }
    try {
        return rrtpff::runBenchmark(std::filesystem::absolute(argv[2]));
    }
    catch(const std::exception& e) {
        std::cerr<<"Error: "<<e.what()<<'\n';
        return 1;
    }
}
