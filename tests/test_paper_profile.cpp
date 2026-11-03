#include "tests/test_support.h"
#include <filesystem>
#include <yaml-cpp/yaml.h>
#include "systems/robot_factory.h"
using namespace rrtpff;
int main(int argc, char **argv) {
    return runTest([&] {
        require(argc == 2, "source directory required");
        const std::filesystem::path root = argv[1];
       
    });
}
