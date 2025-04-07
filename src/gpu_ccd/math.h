#ifndef MATH_HPP
#define MATH_HPP

#include "cubic.h"
#include "types.h"
#include "typedef.h"
#include "array.h"
#include "vector_type_t.h"
namespace cuccd {
    /// @brief Compute the sign of a number.
    /// @param x Number to compute the sign of.
    /// @return -1 if the number is negative, 1 if the number is positive, 0 otherwise.
    CUDA_INLINE_CALLABLE int sgn(real x) { return (real(0) < x) - (x < real(0)); }

    /// @brief Compute the roots of a quadratic equation.
    ///
    /// The roots are sorted in ascending order. Undefined behavior if the equation
    /// has no real roots. If the equation has a real root, the root is repeated.
    ///
    /// @param a Coefficient of the quadratic term.
    /// @param b Coefficient of the linear term.
    /// @param c Constant term.
    /// @return Roots of the quadratic equation.
    CUDA_INLINE_CALLABLE cuccd::array<real, 2>
    solve_quadratic_equation(const real a, const real b, const real c);

    /// @brief Perform a modified Newton-Raphson root finding algorithm to find the roots of a cubic equation.
    /// @param f Cubic equation to find the roots of.
    /// @param x0 Initial guess for the root.
    /// @param tolerance Tolerance for the root finding algorithm.
    /// @return Root of the cubic equation.
    CUDA_INLINE_CALLABLE real newton_raphson(
        const CubicEquation& f, const real x0, const real tolerance = CMP_EPSILON);

    /// @brief Perform a Newton-Raphson root finding algorithm to find the roots of a cubic equation.
    /// @param f Cubic equation to find the roots of.
    /// @param x0 Initial guess for the root.
    /// @param tolerance Tolerance for the root finding algorithm.
    /// @return Root of the cubic equation.
    CUDA_INLINE_CALLABLE real modified_newton_raphson(
        const CubicEquation& f,
        const real x0,
        const real locally_min_gradient,
        const real tolerance = CMP_EPSILON);

    CUDA_INLINE_CALLABLE bool is_point_inside_triangle(
        const point& p,
        const point& t0,
        const point& t1,
        const point& t2);

    CUDA_INLINE_CALLABLE bool are_edges_intersecting(
        const point& ea0,
        const point& ea1,
        const point& eb0,
        const point& eb1);

} // namespace cuccd

namespace cuccd{

    CUDA_INLINE_CALLABLE cuccd::array<real, 2>
    solve_quadratic_equation(const real a, const real b, const real c)
    {
        assert(b * b - 4 * a * c >= 0);
        const real tmp = b + sgn(b) * sqrt(b * b - 4 * a * c);
        cuccd::array<real, 2> roots = { { -2 * c / tmp, -tmp / (2 * a) } };
        if (roots[0] > roots[1]) {
            real tmp = roots[0];
            roots[0] = roots[1];
            roots[1] = tmp;
        }
        return roots;
    }

    CUDA_INLINE_CALLABLE real
    newton_raphson(const CubicEquation& f, const real x0, const real tolerance)
    {
        real prev_x, x = x0;
        do {
            prev_x = x;
            x -= clamp(f(x) / f.derivative(x), real(-1.0), real(1.0));
        } while (abs(x - prev_x) > tolerance);
        return x;
    }

    CUDA_INLINE_CALLABLE real modified_newton_raphson(
        const CubicEquation& f,
        const real x0,
        const real locally_min_gradient,
        const real tolerance)
    {
        real prev_x, x = x0;
        do {
            prev_x = x;
            x -= clamp(f(x) / locally_min_gradient, real(-1.0), real(1.0));
        } while (abs(x - prev_x) > tolerance);
        return x;
    }

    CUDA_INLINE_CALLABLE bool is_point_inside_triangle(
        const point& p,
        const point& t0,
        const point& t1,
        const point& t2)
    {
    
        const point edge0 = t1 - t0;
        const point edge1 = t2 - t0;
        const point p_t0 = p - t0;
        const real dot00 = dot(edge0, edge0);
        const real dot01 = dot(edge0, edge1);
        const real dot11 = dot(edge1, edge1);
        const real dot0p = dot(edge0, p_t0);
        const real dot1p = dot(edge1, p_t0);
        const real denom = dot00 * dot11 - dot01 * dot01;
        const real u = (dot11 * dot0p - dot01 * dot1p) / denom;
        const real v = (dot00 * dot1p - dot01 * dot0p) / denom;
        return u >= 0 && v >= 0 && u + v <= 1;
    }

    CUDA_INLINE_CALLABLE bool are_edges_intersecting(
        const point& ea0,
        const point& ea1,
        const point& eb0,
        const point& eb1)
    {
        const point eb_to_ea = ea0 - eb0;
        const point ea = ea1 - ea0;
        const point eb = eb1 - eb0;

        const real a00 = dot(ea, ea);
        const real a01 = -dot(ea, eb);
        const real a10 = a01;
        const real a11 = dot(eb, eb);

        const real b0 = -dot(eb_to_ea, ea);
        const real b1 = dot(eb_to_ea, eb);

        // Solving the system using Cramer's rule
        const real det = a00 * a11 - a01 * a10;
        if (abs(det) < 1e-10) {
            return false;
        }

        const real s = (a11 * b0 - a01 * b1) / det;
        const real t = (-a10 * b0 + a00 * b1) / det;
        return 0 <= s && s <= 1 && 0 <= t && t <= 1;
    }

} // namespace cuccd


#endif // MATH_HPP
