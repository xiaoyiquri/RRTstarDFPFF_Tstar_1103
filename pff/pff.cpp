#include "pff/pff.h"
// The algorithm lives in the header to retain compile-time system dimensions.
// One implementation serves every Dim/Order instantiation.
template class rrtpff::PFFSolver<2,2>;
template class rrtpff::PFFSolver<2,4>;
template class rrtpff::PFFSolver<4,4>;
