#pragma once
#ifndef RRTPFF_DOUBLE_REFERENCE_TEST_ONLY
#error "double_reference is reserved for isolated test programs"
#endif
#include "pff/pff.h"
namespace rrtpff::reference {
template<int Dim> using DoubleR2 = PFFSolver<Dim,2>;
}
