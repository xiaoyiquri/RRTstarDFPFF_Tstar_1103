#pragma once
#include <iostream>
#include <string_view>
namespace rrtpff {
inline void log(std::string_view message) {std::cerr<<message<<'\n';
}
}
