#include "cubic.h"
#include "math.h"

#include <algorithm>
#include <cassert>

namespace cuccd {


CUDA_INLINE_CALLABLE bool CubicEquation::is_nearly_quadratic(const real tol) const
{
    return abs(a) < tol || abs(a / (b != 0 ? b : 1)) < tol;
}

CUDA_INLINE_CALLABLE bool CubicEquation::is_nearly_linear(const real tol) const
{
    return is_nearly_quadratic()
        && (abs(b) < tol || abs(b / (c != 0 ? c : 1)) < tol);
}

CUDA_INLINE_CALLABLE bool CubicEquation::is_nearly_constant(const real tol) const
{
    return is_nearly_linear()
        && (abs(c) < tol || abs(c / (d != 0 ? d : 1)) < tol);
}

} // namespace ccd