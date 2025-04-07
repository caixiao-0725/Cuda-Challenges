#ifndef CUBIC_HPP
#define CUBIC_HPP

#include <array>
#include <vector>
#include <limits>
#include <iostream>
#include "types.h"
#include "typedef.h"
namespace cuccd {

    /// @brief Cubic equation of the form ax³ + bx² + cx + d.
    struct CubicEquation {
        /// @brief Coefficients of the cubic equation.
        real a, b, c, d;

        /// @brief Evaluate the cubic equation at t.
        /// @param t Value of t to evaluate the cubic equation at.
        /// @return Value of the cubic equation at t.
        CUDA_INLINE_CALLABLE real operator()(const real x) const
        {
            return x * (x * (x * a + b) + c) + d;
        }

        /// @brief Evaluate the derivative of the cubic equation at t.
        /// @param t Value of t to evaluate the derivative of the cubic equation at.
        /// @return Value of the derivative of the cubic equation at t.
        CUDA_INLINE_CALLABLE real derivative(const real x) const
        {
            return x * (x * 3 * a + 2 * b) + c;
        }
        CUDA_INLINE_CALLABLE real inflection() const { return -b / (3 * a); }

        CUDA_INLINE_CALLABLE bool is_nearly_quadratic(
            const real tol = 10 * CMP_EPSILON) const;
        CUDA_INLINE_CALLABLE bool is_nearly_linear(
            const real tol = 10 * CMP_EPSILON) const;
        CUDA_INLINE_CALLABLE bool is_nearly_constant(
            const real tol = 10 * CMP_EPSILON) const;

        CUDA_INLINE_CALLABLE CubicEquation& operator*=(const real x)
        {
            this->a *= x;
            this->b *= x;
            this->c *= x;
            this->d *= x;
            return *this;
        }
    };

} // namespace cuccd
#endif // CUBIC_HPP