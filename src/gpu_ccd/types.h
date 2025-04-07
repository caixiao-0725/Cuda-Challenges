#ifndef TYPES_H
#define TYPES_H

#include <cstddef>
namespace cuccd {
    #ifdef USE_FLOAT
    using real = float;
    using point = float3;
    #else
    using real = double;
    using point = double3;
    #endif
    using root = real[4];
    #define CMP_EPSILON 1e-10

}

#endif
