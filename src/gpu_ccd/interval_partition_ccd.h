#ifndef INTERVAL_PARTITION_CCD_H
#define INTERVAL_PARTITION_CCD_H
#include "math.h"
#include "autogen.h"
#include "array.h"
#include "types.h"
#include "cyPolynomial.h"
#include "vector_type_t.h"

namespace cuccd {
    class IntervalPartitionCCD {
       using point = cuccd::point;
       using real = cuccd::real;
    public:
        //find the roots of the cubic equation in the interval [0, 1], using Yuksel, 2022 method (public release codebase).
        static CUDA_INLINE_CALLABLE int interval_partition_root_finder(const CubicEquation& cubic_equation, cuccd::root& roots);
        static CUDA_INLINE_CALLABLE bool point_triangle_ccd(const point& p_t0, const point& t0_t0, const point& t1_t0, const point& t2_t0, const point& p_t1, const point& t0_t1, const point& t1_t1, const point& t2_t1, real& toi);
        static CUDA_INLINE_CALLABLE bool edge_edge_ccd(const point& ea0_t0, const point& ea1_t0, const point& eb0_t0, const point& eb1_t0, const point& ea0_t1, const point& ea1_t1, const point& eb0_t1, const point& eb1_t1, real& toi);
    };
}

namespace cuccd {

    CUDA_INLINE_CALLABLE int IntervalPartitionCCD::interval_partition_root_finder(const CubicEquation& cubic_equation, cuccd::root& roots) {
        double coef[4] = {cubic_equation.d, cubic_equation.c, cubic_equation.b, cubic_equation.a};
        double roots_array[3];
        int root_current = 0;
        // Use cyPolynomial's cubic root finder
        int num_roots = cy::CubicRoots(roots_array, coef, 0.0, 1.0);
        // Add valid roots to the result vector
        for (int i = 0; i < num_roots; i++) {
            if (roots_array[i] >= 0.0 && roots_array[i] <= 1.0) {
                roots[root_current] = roots_array[i];
                ++root_current;
            }
        }
        
        return root_current;
    }

    CUDA_INLINE_CALLABLE bool IntervalPartitionCCD::point_triangle_ccd(const point& p_t0, const point& t0_t0, const point& t1_t0, const point& t2_t0,
                                                 const point& p_t1, const point& t0_t1, const point& t1_t1, const point& t2_t1,
                                                 real& toi) {
        // Generate the cubic equation using autogen
        auto eq = autogen::point_triangle_ccd_equation(
            p_t0.x, p_t0.y, p_t0.z,
            t0_t0.x, t0_t0.y, t0_t0.z,
            t1_t0.x, t1_t0.y, t1_t0.z,
            t2_t0.x, t2_t0.y, t2_t0.z,
            p_t1.x, p_t1.y, p_t1.z,
            t0_t1.x, t0_t1.y, t0_t1.z,
            t1_t1.x, t1_t1.y, t1_t1.z,
            t2_t1.x, t2_t1.y, t2_t1.z);

        // Find roots using interval partition method
        cuccd::root roots;
        roots[0] = -1; roots[1] = -1; roots[2] = -1; roots[3] = -1;
        auto num_roots = interval_partition_root_finder(eq, roots);
        
        for (int i = 0; i < num_roots; i++) {
            real root = roots[i];
            point p_interp = lerp(p_t0, p_t1, root);
            point t0_interp = lerp(t0_t0, t0_t1, root);
            point t1_interp = lerp(t1_t0, t1_t1, root);
            point t2_interp = lerp(t2_t0, t2_t1, root);
    
            if (is_point_inside_triangle(p_interp, t0_interp, t1_interp, t2_interp)) {
                toi = root;
                return true;
            }
        }
        
        return false;
    }

    CUDA_INLINE_CALLABLE bool IntervalPartitionCCD::edge_edge_ccd(const point& ea0_t0, const point& ea1_t0, const point& eb0_t0, const point& eb1_t0,
                                            const point& ea0_t1, const point& ea1_t1, const point& eb0_t1, const point& eb1_t1,
                                            real& toi) {
        // Generate the cubic equation using autogen
        auto eq = autogen::edge_edge_ccd_equation(
            ea0_t0.x, ea0_t0.y, ea0_t0.z,
            ea1_t0.x, ea1_t0.y, ea1_t0.z,
            eb0_t0.x, eb0_t0.y, eb0_t0.z,
            eb1_t0.x, eb1_t0.y, eb1_t0.z,
            ea0_t1.x, ea0_t1.y, ea0_t1.z,
            ea1_t1.x, ea1_t1.y, ea1_t1.z,
            eb0_t1.x, eb0_t1.y, eb0_t1.z,
            eb1_t1.x, eb1_t1.y, eb1_t1.z);
        cuccd::root roots;
        roots[0] = -1; roots[1] = -1; roots[2] = -1; roots[3] = -1;
        auto num_roots = interval_partition_root_finder(eq, roots);
        

        for (int i = 0; i < num_roots; i++) {
            real root = roots[i];
            point ea0_interp = lerp(ea0_t0, ea0_t1, root);
            point ea1_interp = lerp(ea1_t0, ea1_t1, root);
            point eb0_interp = lerp(eb0_t0, eb0_t1, root);
            point eb1_interp = lerp(eb1_t0, eb1_t1, root);
            if (are_edges_intersecting(ea0_interp, ea1_interp, eb0_interp, eb1_interp)) {
                toi = root;
                return true;
            }
        }
        
        return false;
    }
}



#endif