#ifndef DIRECT_CUBIC_ROOT_FINDER_CCD_H
#define DIRECT_CUBIC_ROOT_FINDER_CCD_H

#include "cubic.h"
#include "autogen.h"
#include "math.h"
#include "types.h" 
#include <vector>
#include "vector_type_t.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace cuccd {
    class DirectCubicRootFinder {
        using point = cuccd::point;
    public:
        // This method will find the roots of the cubic equation in the interval [0, 1]
        // Uses a combination of Cardano's method and trigonometric solutions
        // Returns conservative estimates suitable for CCD applications
        CUDA_INLINE_CALLABLE static int find_roots(const CubicEquation& cubic_equation, cuccd::root& roots);
        CUDA_INLINE_CALLABLE static int find_roots_none_conservative(const CubicEquation& cubic_equation, cuccd::root& roots);
        // CCD functions: will generate the cubic equation and then call the root finder, if the root is in the interval [0, 1] then do point inside triangle or edge edge intersection test for validation.
        CUDA_INLINE_CALLABLE static bool point_triangle_ccd(const point& p_t0, const point& t0_t0, const point& t1_t0, const point& t2_t0, const point& p_t1, const point& t0_t1, const point& t1_t1, const point& t2_t1, real& toi);
        CUDA_INLINE_CALLABLE static bool edge_edge_ccd(const point& ea0_t0, const point& ea1_t0, const point& eb0_t0, const point& eb1_t0, const point& ea0_t1, const point& ea1_t1, const point& eb0_t1, const point& eb1_t1, real& toi);
    
    private:
        // Helper methods for conservative root refinement
        CUDA_INLINE_CALLABLE static bool is_zero(real val);
        CUDA_INLINE_CALLABLE static int  count_crossings(const CubicEquation& cubic, real start, real end);
        CUDA_INLINE_CALLABLE static real find_earliest_crossing(const CubicEquation& cubic, real start, real end);
        CUDA_INLINE_CALLABLE static real conservative_binary_search(const CubicEquation& cubic, real start, real end);
        CUDA_INLINE_CALLABLE static real evaluate_derivative(const CubicEquation& cubic, real x);
        CUDA_INLINE_CALLABLE static real find_max_gradient_point(const CubicEquation& cubic, real start, real end);
        CUDA_INLINE_CALLABLE static real refine_conservative_root(const CubicEquation& cubic, real initialRoot, bool isAlreadyConservative);
        CUDA_INLINE_CALLABLE static void refine_roots_conservatively(const CubicEquation& cubic, cuccd::root& roots);   
        // Sort roots array from small to large, keeping any -1 values at the end
        CUDA_INLINE_CALLABLE static void sort4(cuccd::root& roots);
    };
}

namespace cuccd {

// Define comparison epsilon for floating point comparisons

    CUDA_INLINE_CALLABLE bool DirectCubicRootFinder::is_zero(real val) {
        return abs(val) < CMP_EPSILON;
    }

    // Count the number of zero crossings in an interval
    CUDA_INLINE_CALLABLE int DirectCubicRootFinder::count_crossings(const CubicEquation& cubic, real start, real end) {
        constexpr int SAMPLES = 20; // Number of samples to check
        int count = 0;
        real prevF = cubic(start);
        
        for (int i = 1; i <= SAMPLES; i++) {
            real t = start + (end - start) * i / SAMPLES;
            real f = cubic(t);
            
            if (prevF * f < 0 || is_zero(f)) {
                count++;
            }
            
            prevF = f;
        }
        
        return count;
    }
    
    // Find the earliest zero crossing in an interval
    CUDA_INLINE_CALLABLE real DirectCubicRootFinder::find_earliest_crossing(const CubicEquation& cubic, real start, real end) {
        constexpr int MAX_ITERATIONS = 30;
        constexpr real CONVERGENCE = CMP_EPSILON * 0.1;
        
        real a = start;
        real b = end;
        real fa = cubic(a);
        
        // Binary search for the first crossing
        for (int i = 0; i < MAX_ITERATIONS; i++) {
            real mid = (a + b) * 0.5;
            real fmid = cubic(mid);
            
            if (is_zero(fmid)) {
                // Found exact root - move slightly left to be conservative
                return mid - CONVERGENCE;
            }
            
            if (fa * fmid < 0) {
                // Zero crossing in [a, mid]
                b = mid;
            } else {
                // Zero crossing in [mid, b]
                a = mid;
                fa = fmid;
            }
            
            if (b - a < CONVERGENCE) {
                // Converged - return slightly before the crossing
                return a;
            }
        }
        
        // Return conservative value if not converged
        return a;
    }
    
    // Conservative binary search
    CUDA_INLINE_CALLABLE real DirectCubicRootFinder::conservative_binary_search(const CubicEquation& cubic, real start, real end) {
        constexpr int MAX_ITERATIONS = 20;
        
        real a = start;
        real b = end;
        real fa = cubic(a);
        real fb = cubic(b);
        
        // Handle special cases
        if (is_zero(fa)) return start;
        if (is_zero(fb)) return end - CMP_EPSILON;
        
        // If no sign change, return the endpoint closest to zero
        if (fa * fb > 0) {
            return (abs(fa) < abs(fb)) ? start : end;
        }
        
        // Binary search with conservative bias
        for (int i = 0; i < MAX_ITERATIONS; i++) {
            real mid = 0.4*a + 0.6*b; // Biased toward the lower end
            real fmid = cubic(mid);
            
            if (is_zero(fmid)) {
                return mid - CMP_EPSILON;
            }
            
            if (fa * fmid < 0) {
                b = mid;
                fb = fmid;
            } else {
                a = mid;
                fa = fmid;
            }
            
            if (b - a < CMP_EPSILON) {
                return a; // Conservative choice
            }
        }
        
        return a; // Most conservative estimate
    }
    
    // Compute the derivative at a point
    CUDA_INLINE_CALLABLE real DirectCubicRootFinder::evaluate_derivative(const CubicEquation& cubic, real x) {
        return real(3.0) * cubic.a * x * x + real(2.0) * cubic.b * x + cubic.c;
    }
    
    // Find point with maximum gradient in interval
    CUDA_INLINE_CALLABLE real DirectCubicRootFinder::find_max_gradient_point(const CubicEquation& cubic, real start, real end) {
        // For a cubic function: f'(x) = 3ax² + 2bx + c
        // The extremum of f' is where f''(x) = 0, so where 6ax + 2b = 0
        // Therefore, x = -b/(3a)
        
        real inflectionPoint = -cubic.b / (real(3.0) * cubic.a);
        
        // Check if the inflection point is within our interval
        if (inflectionPoint > start && inflectionPoint < end) {
            // Compute the second derivative value to determine if it's a max or min
            real secondDeriv = real(6.0) * cubic.a;
            
            if (secondDeriv < 0) {
                // If negative, it's a maximum of the first derivative
                return inflectionPoint;
            }
        }
        
        // If we get here, either the inflection point is outside our interval
        // or it's a minimum of the derivative
        // Check the derivative at both endpoints
        real derivStart = evaluate_derivative(cubic, start);
        real derivEnd = evaluate_derivative(cubic, end);
        
        // Return the point with the larger absolute derivative
        return (abs(derivStart) > abs(derivEnd)) ? start : end;
    }
    
    // Refined method for conservative root approximation
    CUDA_INLINE_CALLABLE real DirectCubicRootFinder::refine_conservative_root(const CubicEquation& cubic, real initialRoot, bool isAlreadyConservative) {
        constexpr int MAX_ITERATIONS = 15;
        real root = initialRoot;
        real f0 = cubic(0);
        
        if (isAlreadyConservative) {
            // Already on the conservative side - use regular Newton with safety checks
            for (int i = 0; i < MAX_ITERATIONS; i++) {
                real f = cubic(root);
                real fprime = evaluate_derivative(cubic, root);
                
                if (is_zero(fprime)) break;
                
                real newRoot = root - f / fprime;
                
                // Verify we're still on the conservative side
                real fNew = cubic(newRoot);
                if (f0 * fNew <= 0) {
                    // Jumped to the non-conservative side - revert and exit
                    break;
                }
                if (abs(newRoot - root) < CMP_EPSILON) break;
                
                root = newRoot;
            }
        } else {
            // Not on conservative side - use maximum gradient approach
            real left = 0.0;
            real right = root;
            
            for (int i = 0; i < MAX_ITERATIONS; i++) {
                // Find the maximum gradient point in the interval
                real maxGradPoint = find_max_gradient_point(cubic, left, right);
                real maxGradient = abs(evaluate_derivative(cubic, maxGradPoint));

                real f = cubic(root);
                if (is_zero(f)) {
                    return root;
                }
                
                // Conservative step using maximum gradient
                real step = f / maxGradient;
                real newRoot = root - step;
                
                // Keep in bounds
                newRoot = max(left, min(right, newRoot));
                
                real fNew = cubic(newRoot);
                
                if (f0 * fNew > 0) {
                    // Now on conservative side - do a final regular Newton step
                    real fprime = evaluate_derivative(cubic, newRoot);
                    if (!is_zero(fprime)) {
                        real finalStep = fNew / fprime;
                        real candidateRoot = newRoot - finalStep;
                        
                        // Only accept if still on conservative side
                        real fCandidate = cubic(candidateRoot);
                        if (f0 * fCandidate > 0) {
                            newRoot = candidateRoot;
                        }
                    }
                    return newRoot;
                }
                
                // Update binary search interval
                right = newRoot;

                if (abs(newRoot - root) < CMP_EPSILON || right - left < CMP_EPSILON) {
                    // Not yet on conservative side - use binary search for final approach
                    return conservative_binary_search(cubic, left, right);
                }
                
                root = newRoot;
            }
            
            // If we've exhausted iterations and still not on conservative side, use binary search as a fallback
            return conservative_binary_search(cubic, 0.0, initialRoot);
        }
        
        return root;
    }
    
    
    //refine roots conservatively for CCD
    CUDA_INLINE_CALLABLE void DirectCubicRootFinder::refine_roots_conservatively(const CubicEquation& cubic, cuccd::root& roots) { 
        for (int i = 0; i < 4; i++) {
            real root = roots[i];
            //invalid roots will be skipped
            if (root < 0) {
                continue;
            }
            if (abs(root - 1.0) < CMP_EPSILON) {
                root = min(root, real(1) - CMP_EPSILON);
                roots[i] = root;
                continue;
            }

            real f0 = cubic(0);
            real fRoot = cubic(root);
            
            if (is_zero(fRoot)) {
                root = root - CMP_EPSILON;
                roots[i] = root;
                continue;
            }
            
            // Handle multiple crossings by checking sign changes between 0 and root
            int crossingCount = count_crossings(cubic, 0.0, root);
            
            if (crossingCount == 0) {
                // No crossings found - numerical error or tangential touch
                // Use binary search to find a conservative approximation
                root = conservative_binary_search(cubic, 0.0, root);
                roots[i] = root;
                continue;
            } else if (crossingCount > 1) {
                // Multiple crossings - find the earliest one
                root = find_earliest_crossing(cubic, 0.0, root);
                roots[i] = root;
                continue;
            }
            
            // Single crossing case - determine if root is conservative
            bool isRootConservative = (f0 * fRoot > 0);
            
            if (isRootConservative) {
                // Already conservative - refine for accuracy
                root = refine_conservative_root(cubic, root, true);
            } else {
                // Not conservative - need to find a conservative approximation
                root = refine_conservative_root(cubic, root, false);
            }
            
            root = max(real(0), min(root, real(1)));
            roots[i] = root;
        }
        
        // Sort the roots
        sort4(roots);
        
        // Remove duplicates while preserving -1 values at the end
        for (int i = 1; i < 4; i++) {
            if (roots[i] > 0 && roots[i-1] > 0 && is_zero(roots[i] - roots[i-1])) {
                // Shift all subsequent roots
                for (int j = i; j < 3; j++) {
                    roots[j] = roots[j+1];
                }
                roots[3] = -1; // Make sure the last root is -1
            }
        }
    }

    CUDA_INLINE_CALLABLE void DirectCubicRootFinder::sort4(cuccd::root& roots) {
        // Insertion sort for 4 elements, keeping -1 at the end
        // First, count valid roots (not equal to -1)
        int valid_count = 0;
        for (int i = 0; i < 4; i++) {
            if (roots[i] != -1) valid_count++;
        }

        // Sort only the valid roots
        for (int i = 1; i < valid_count; i++) {
            real key = roots[i];
            int j = i - 1;
            
            // Move elements that are greater than key to one position ahead
            while (j >= 0 && roots[j] > key) {
                roots[j + 1] = roots[j];
                j--;
            }
            roots[j + 1] = key;
        }
        
        // Ensure any -1 values are at the end
        if (valid_count < 4) {
            // Collect valid roots first
            real valid_roots[4];
            int valid_idx = 0;
            
            for (int i = 0; i < 4; i++) {
                if (roots[i] != -1) {
                    valid_roots[valid_idx++] = roots[i];
                }
            }
            
            // Place valid roots at the beginning
            for (int i = 0; i < valid_count; i++) {
                roots[i] = valid_roots[i];
            }
            
            // Fill the rest with -1
            for (int i = valid_count; i < 4; i++) {
                roots[i] = -1;
            }
        }
    }

    CUDA_INLINE_CALLABLE int DirectCubicRootFinder::find_roots(const CubicEquation& cubic_equation, cuccd::root& roots) {
        real a = cubic_equation.a;
        real b = cubic_equation.b;
        real c = cubic_equation.c;
        real d = cubic_equation.d;
        roots[0] = -1; roots[1] = -1; roots[2] = -1; roots[3] = -1;
        // Special case: check endpoints
        real val0 = cubic_equation(0);
        real val1 = cubic_equation(1);
        int current_root_index = 0;
        if (is_zero(val0)) roots[current_root_index++]=0;
        if (is_zero(val1) && (current_root_index == 0 || !is_zero(1 - roots[current_root_index-1]))) roots[current_root_index++]=1;
        
        // Handle the case where the equation is not cubic
        if (is_zero(a)) {
            // Handle quadratic case
            if (!is_zero(b)) {
                // b*x^2 + c*x + d = 0
                real discriminant = c*c - 4*b*d;
                if (discriminant >= 0) {
                    real sqrtDiscr = sqrt(discriminant);
                    real root1 = (-c + sqrtDiscr) / (2*b);
                    real root2 = (-c - sqrtDiscr) / (2*b);
                    
                    if (root1 >= 0 && root1 <= 1 && current_root_index < 4) roots[current_root_index++] = root1;
                    if (root2 >= 0 && root2 <= 1 && current_root_index < 4) roots[current_root_index++] = root2;
                }
            } else if (!is_zero(c)) {
                // Linear case: c*x + d = 0
                real root = -d / c;
                if (root >= 0 && root <= 1 && current_root_index < 4) roots[current_root_index++] = root;
            } else if (is_zero(d)) {
                // The equation is 0 = 0, so every point is a root, just add a point in the middle
                if (current_root_index < 4) roots[current_root_index++] = 0.5;
            }
            
            // Sort the roots
            sort4(roots);
            
            // Apply conservative refinement to the roots
            refine_roots_conservatively(cubic_equation, roots);
            
            return current_root_index;
        }
        
        // For actual cubic equations, based on Cardano's method and trigonometric solutions
        
        // First, check if we can factor out an x
        if (is_zero(d)) {
            // The equation is x(ax^2 + bx + c) = 0
            // So x=0 is a root (already checked above)
            
            // Solve the quadratic part
            real discr = b*b - 4*a*c;
            if (discr >= 0) {
                real sqrtDiscr = sqrt(discr);
                real root1 = (-b + sqrtDiscr) / (2*a);
                real root2 = (-b - sqrtDiscr) / (2*a);
                
                if (root1 > 0 && root1 < 1) roots[current_root_index++] = root1;
                if (root2 > 0 && root2 < 1) roots[current_root_index++] = root2;
            }
            
            // Sort roots
            sort4(roots);
            
            // Apply conservative refinement to the roots
            refine_roots_conservatively(cubic_equation, roots);
            
            return current_root_index;
        }
        
        // Convert to depressed cubic form: t^3 + p*t + q = 0
        // by substituting x = t - b/(3*a)
        real p = (3*a*c - b*b) / (3*a*a);
        real q = (2*b*b*b - 9*a*b*c + 27*a*a*d) / (27*a*a*a);
        
        // Calculate the discriminant
        real delta = (q*q/4) + (p*p*p/27);
        
        // Offset for converting back to original variable
        real offset = b / (3*a);
        
        if (is_zero(delta)) {
            // Case: delta = 0, special case with repeated roots
            if (is_zero(p)) {
                // Triple root case
                real root = -offset;
                if (root > 0 && root < 1) roots[current_root_index++] = root;
            } else {
                // One single root and one double root
                real t1 = 3*q/p;
                real t2 = -3*q/(2*p);
                
                real root1 = t1 - offset;
                real root2 = t2 - offset;
                
                if (root1 >= 0 && root1 <= 1) roots[current_root_index++] = root1;
                if (root2 >= 0 && root2 <= 1) roots[current_root_index++] = root2;
            }
        } else if (delta > 0) {
            // Case: delta > 0, one real root
            real u = cbrt(-q/2 + sqrt(delta));
            real v = cbrt(-q/2 - sqrt(delta));
            
            real t = u + v;
            real root = t - offset;
            
            if (root >= 0 && root <= 1) roots[current_root_index++] = root;
        } else {
            // Case: delta < 0, three real roots using trigonometric solution
            real rho = sqrt(-p*p*p / 27);
            real theta = acos(-q / (2*rho));
            
            for (int k = 0; k < 3; k++) {
                real t = 2 * sqrt(-p/3) * cos((theta + 2*k*M_PI) / 3);
                real root = t - offset;
                
                if (root >= 0 && root <= 1) {
                    // Check for duplicates within epsilon
                    bool isDuplicate = false;
                    for (real existingRoot : roots) {
                        if (abs(existingRoot - root) < CMP_EPSILON) {
                            isDuplicate = true;
                            break;
                        }
                    }
                    if (!isDuplicate) roots[current_root_index++] = root;
                }
            }
        }
        
        // At the end, replace std::sort calls with sort4
        sort4(roots);
        
        // Apply conservative refinement
        refine_roots_conservatively(cubic_equation, roots);
        
        return current_root_index;
    }

    CUDA_INLINE_CALLABLE int DirectCubicRootFinder::find_roots_none_conservative(const CubicEquation& cubic_equation, cuccd::root& roots) {
        real a = cubic_equation.a;
        real b = cubic_equation.b;
        real c = cubic_equation.c;
        real d = cubic_equation.d;
        
        // Initialize the roots
        roots[0] = -1; roots[1] = -1; roots[2] = -1; roots[3] = -1;
        int current_root_index = 0;

        // Special case: check endpoints
        real val0 = cubic_equation(0);
        real val1 = cubic_equation(1);
        
        if (is_zero(val0)) roots[current_root_index++] = 0;
        if (is_zero(val1) && (current_root_index == 0 || !is_zero(1 - roots[current_root_index-1]))) 
            roots[current_root_index++] = 1;
        
        // Handle the case where the equation is not cubic
        if (is_zero(a)) {
            // Handle quadratic case
            if (!is_zero(b)) {
                // b*x^2 + c*x + d = 0
                real discriminant = c*c - 4*b*d;
                if (discriminant >= 0) {
                    real sqrtDiscr = sqrt(discriminant);
                    real root1 = (-c + sqrtDiscr) / (2*b);
                    real root2 = (-c - sqrtDiscr) / (2*b);
                    
                    if (root1 >= 0 && root1 <= 1 && current_root_index < 4) 
                        roots[current_root_index++] = root1;
                    if (root2 >= 0 && root2 <= 1 && current_root_index < 4 && !is_zero(root2 - root1)) 
                        roots[current_root_index++] = root2;
                }
            } else if (!is_zero(c)) {
                // Linear case: c*x + d = 0
                real root = -d / c;
                if (root >= 0 && root <= 1 && current_root_index < 4) 
                    roots[current_root_index++] = root;
            } else if (is_zero(d)) {
                // The equation is 0 = 0, so every point is a root, just add a point in the middle
                if (current_root_index < 4) 
                    roots[current_root_index++] = 0.5;
            }
            
            // Sort roots
            sort4(roots);
            
            return current_root_index;
        }
        
        // For actual cubic equations
        
        // First, check if we can factor out an x
        if (is_zero(d)) {
            // The equation is x(ax^2 + bx + c) = 0
            // So x=0 is a root (already checked above)
            
            // Solve the quadratic part
            real discr = b*b - 4*a*c;
            if (discr >= 0) {
                real sqrtDiscr = sqrt(discr);
                real root1 = (-b + sqrtDiscr) / (2*a);
                real root2 = (-b - sqrtDiscr) / (2*a);
                
                if (root1 > 0 && root1 < 1 && current_root_index < 4) 
                    roots[current_root_index++] = root1;
                if (root2 > 0 && root2 < 1 && current_root_index < 4 && !is_zero(root2 - root1)) 
                    roots[current_root_index++] = root2;
            }
            
            // Sort roots
            sort4(roots);
            
            return current_root_index;
        }
        
        // Convert to depressed cubic form: t^3 + p*t + q = 0
        // by substituting x = t - b/(3*a)
        real p = (3*a*c - b*b) / (3*a*a);
        real q = (2*b*b*b - 9*a*b*c + 27*a*a*d) / (27*a*a*a);
        
        // Calculate the discriminant
        real delta = (q*q/4) + (p*p*p/27);
        
        // Offset for converting back to original variable
        real offset = b / (3*a);
        
        if (is_zero(delta)) {
            // Case: delta = 0, special case with repeated roots
            if (is_zero(p)) {
                // Triple root case
                real root = -offset;
                if (root > 0 && root < 1 && current_root_index < 4) 
                    roots[current_root_index++] = root;
            } else {
                // One single root and one double root
                real t1 = 3*q/p;
                real t2 = -3*q/(2*p);
                
                real root1 = t1 - offset;
                real root2 = t2 - offset;
                
                if (root1 >= 0 && root1 <= 1 && current_root_index < 4) 
                    roots[current_root_index++] = root1;
                if (root2 >= 0 && root2 <= 1 && current_root_index < 4 && !is_zero(root2 - root1)) 
                    roots[current_root_index++] = root2;
            }
        } else if (delta > 0) {
            // Case: delta > 0, one real root
            real u = cbrt(-q/2 + sqrt(delta));
            real v = cbrt(-q/2 - sqrt(delta));
            
            real t = u + v;
            real root = t - offset;
            
            if (root >= 0 && root <= 1 && current_root_index < 4) 
                roots[current_root_index++] = root;
        } else {
            // Case: delta < 0, three real roots using trigonometric solution
            real rho = sqrt(-p*p*p / 27);
            real theta = acos(-q / (2*rho));
            
            for (int k = 0; k < 3 && current_root_index < 4; k++) {
                real t = 2 * sqrt(-p/3) * cos((theta + 2*k*M_PI) / 3);
                real root = t - offset;
                
                if (root >= 0 && root <= 1) {
                    // Check for duplicates within epsilon
                    bool isDuplicate = false;
                    for (int j = 0; j < current_root_index; j++) {
                        if (abs(roots[j] - root) < CMP_EPSILON) {
                            isDuplicate = true;
                            break;
                        }
                    }
                    if (!isDuplicate) 
                        roots[current_root_index++] = root;
                }
            }
        }
        
        // Sort the roots
        sort4(roots);
        
        return current_root_index;
    }

    CUDA_INLINE_CALLABLE bool DirectCubicRootFinder::point_triangle_ccd(const point& p_t0, const point& t0_t0, const point& t1_t0, const point& t2_t0, const point& p_t1, const point& t0_t1, const point& t1_t1, const point& t2_t1, real& toi) {
        auto eq = 
            autogen::point_triangle_ccd_equation(
                p_t0.x, p_t0.y, p_t0.z, t0_t0.x, t0_t0.y, t0_t0.z,
                t1_t0.x, t1_t0.y, t1_t0.z, t2_t0.x, t2_t0.y, t2_t0.z,
                p_t1.x, p_t1.y, p_t1.z, t0_t1.x, t0_t1.y, t0_t1.z,
                t1_t1.x, t1_t1.y, t1_t1.z, t2_t1.x, t2_t1.y, t2_t1.z);
        
        cuccd::root roots;
        roots[0] = -1; roots[1] = -1; roots[2] = -1; roots[3] = -1;
        int num_roots = find_roots(eq, roots);
        
        if (num_roots == 0) return false;
        
        for (int i = 0; i < num_roots; i++) {
            real root = roots[i];
            if (root < 0) continue; // Skip invalid roots
            
            // Use lerp function for interpolation
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

    CUDA_INLINE_CALLABLE bool DirectCubicRootFinder::edge_edge_ccd(const point& ea0_t0, const point& ea1_t0, const point& eb0_t0, const point& eb1_t0, const point& ea0_t1, const point& ea1_t1, const point& eb0_t1, const point& eb1_t1, real& toi) {
        auto eq = autogen::edge_edge_ccd_equation(
            ea0_t0.x, ea0_t0.y, ea0_t0.z, ea1_t0.x, ea1_t0.y, ea1_t0.z,
            eb0_t0.x, eb0_t0.y, eb0_t0.z, eb1_t0.x, eb1_t0.y, eb1_t0.z,
            ea0_t1.x, ea0_t1.y, ea0_t1.z, ea1_t1.x, ea1_t1.y, ea1_t1.z,
            eb0_t1.x, eb0_t1.y, eb0_t1.z, eb1_t1.x, eb1_t1.y, eb1_t1.z);
        
        cuccd::root roots;
        roots[0] = -1; roots[1] = -1; roots[2] = -1; roots[3] = -1;
        int num_roots = find_roots(eq, roots);
        
        if (num_roots == 0) return false;
        
        for (int i = 0; i < num_roots; i++) {
            real root = roots[i];
            if (root < 0) continue; // Skip invalid roots
            
            // Use lerp function for interpolation
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