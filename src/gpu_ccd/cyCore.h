// cyCodeBase by Cem Yuksel
// [www.cemyuksel.com]
//-------------------------------------------------------------------------------
//! \file   cyCore.h 
//! \author Cem Yuksel
//! 
//! \brief  Core functions and macros
//! 
//! Core functions and macros for math and other common operations
//! 
//-------------------------------------------------------------------------------
//
// Copyright (c) 2016, Cem Yuksel <cem@cemyuksel.com>
// All rights reserved.
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy 
// of this software and associated documentation files (the "Software"), to deal 
// in the Software without restriction, including without limitation the rights 
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell 
// copies of the Software, and to permit persons to whom the Software is 
// furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in all 
// copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE 
// SOFTWARE.
// 
//-------------------------------------------------------------------------------


#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cassert>
#include <cmath>
#include <type_traits>
#include <limits>
#include "typedef.h"

//-------------------------------------------------------------------------------
namespace cy {


// nodiscard
//#if _CY_COMPILER_VER_MEETS(1901,40800,30000,1500)
#if (__cplusplus>=201703L) || (defined(_MSVC_LANG) && _MSVC_LANG>=201703L)
# define CY_NODISCARD [[nodiscard]]
#else
# define CY_NODISCARD
#endif


//!@name Common math function templates

template <typename T> CUDA_INLINE_CALLABLE CY_NODISCARD  T Max      ( T v1, T v2 ) { return v1 >= v2 ? v1 : v2; }
template <typename T> CUDA_INLINE_CALLABLE CY_NODISCARD  T Min      ( T v1, T v2 ) { return v1 <= v2 ? v1 : v2; }
template <typename T> CUDA_INLINE_CALLABLE CY_NODISCARD  T Max      ( T v1, T v2, T v3 ) { return Max( Max(v1,v2), v3 ); }
template <typename T> CUDA_INLINE_CALLABLE CY_NODISCARD  T Min      ( T v1, T v2, T v3 ) { return Min( Min(v1,v2), v3 ); }
template <typename T> CUDA_INLINE_CALLABLE CY_NODISCARD  T Max      ( T v1, T v2, T v3, T const v4 ) { return Max( Max(v1,v2), Max(v3,v4) ); }
template <typename T> CUDA_INLINE_CALLABLE CY_NODISCARD  T Min      ( T v1, T v2, T v3, T const v4 ) { return Min( Min(v1,v2), Min(v3,v4) ); }
template <typename T> CUDA_INLINE_CALLABLE CY_NODISCARD  T Clamp    ( T v, T minVal=T(0), T maxVal=T(1) ) { return Min(maxVal,Max(minVal,v)); }

template <typename T> CUDA_INLINE_CALLABLE CY_NODISCARD  T ACosSafe ( T v ) { return (T) acos(Clamp(v,T(-1),T(1))); }
template <typename T> CUDA_INLINE_CALLABLE CY_NODISCARD  T ASinSafe ( T v ) { return (T) asin(Clamp(v,T(-1),T(1))); }
template <typename T> CUDA_INLINE_CALLABLE CY_NODISCARD  T Sqrt     ( T v ) { return (T) sqrt(v); }
template <typename T> CUDA_INLINE_CALLABLE CY_NODISCARD  T SqrtSafe ( T v ) { return (T) sqrt(Max(v,T(0))); }


template<typename T> CUDA_INLINE_CALLABLE constexpr  T Pi() { return T(3.141592653589793238462643383279502884197169); }

template <typename T>
CUDA_INLINE_CALLABLE CY_NODISCARD __host__ __device__ bool IsFinite(T v) {
    // For integer types, always return true
    if (std::numeric_limits<T>::is_integer) {
        return true;
    }
    // For floating point types, use CUDA's isfinite
    #ifdef __CUDA_ARCH__
        return ::isfinite(v); // CUDA's isfinite in device code
    #else
        return std::isfinite(v); // std::isfinite in host code
    #endif
}

/////////////////////////////////////////////////////////////////////////////////
// Sorting functions
/////////////////////////////////////////////////////////////////////////////////

template <bool ascending, typename T>
CUDA_INLINE_CALLABLE void Sort2( T &r0, T &r1, T const &v0, T const &v1 )
{
	if ( ascending ) {
		r0 = Min( v0, v1 );
		r1 = Max( v0, v1 );
	} else {
		r0 = Max( v0, v1 );
		r1 = Min( v0, v1 );
	}
}

template <bool ascending, typename T>
CUDA_INLINE_CALLABLE void Sort2( T r[2], T const v[2] )
{
	r[1-ascending] = Min( v[0], v[1] );
	r[  ascending] = Max( v[0], v[1] );
}

template <bool ascending, typename T>
CUDA_INLINE_CALLABLE void Sort3( T &r0, T &r1, T &r2, T const &v0, T const &v1, T const &v2 )
{
	T n01   = Min( v0,  v1    );
	T x01   = Max( v0,  v1    );
	T n2x01 = Min( v2,  x01   );
	r1      = Max( n01, n2x01 );
	if ( ascending ) {
		r0  = Min( n2x01, n01 );
		r2  = Max( x01,   v2  );
	} else {
		r0  = Max( x01,   v2  );
		r2  = Min( n2x01, n01 );
	}
}

template <bool ascending, typename T>
CUDA_INLINE_CALLABLE void Sort3( T r[3], T const v[3] )
{
	T n01   = Min( v[0], v[1] );
	T x01   = Max( v[0], v[1] );
	T n2x01 = Min( v[2], x01  );
	T r0    = Min( n2x01, n01 );
	T r1    = Max( n01, n2x01 );
	T r2    = Max( x01,  v[2] );
	if ( ascending ) { r[0]=r0; r[1]=r1; r[2]=r2; }
	else             { r[0]=r2; r[1]=r1; r[2]=r0; }
}

template <bool ascending, typename T>
CUDA_INLINE_CALLABLE void Sort4( T &r0, T &r1, T &r2, T &r3, T const &v0, T const &v1, T const &v2, T const &v3 )
{
	T n01  = Min( v0,  v1  );
	T x01  = Max( v0,  v1  );
	T n23  = Min( v2,  v3  );
	T x23  = Max( v2,  v3  );
	T x02  = Max( n23, n01 );
	T n13  = Min( x01, x23 );
	if ( ascending ) {
		r0 = Min( n01, n23 );
		r1 = Min( x02, n13 );
		r2 = Max( n13, x02 );
		r3 = Max( x23, x01 );
	} else {
		r0 = Max( x23, x01 );
		r1 = Max( n13, x02 );
		r2 = Min( x02, n13 );
		r3 = Min( n01, n23 );
	}
}

template <bool ascending, typename T>
CUDA_INLINE_CALLABLE void Sort4( T r[4], T const v[4] )
{
	T n01 = Min( v[0], v[1] );
	T x01 = Max( v[0], v[1] );
	T n23 = Min( v[2], v[3] );
	T x23 = Max( v[2], v[3] );
	T x02 = Max( n23, n01 );
	T n13 = Min( x01, x23 );
	T r0  = Min( n01, n23 );
	T r1  = Min( x02, n13 );
	T r2  = Max( n13, x02 );
	T r3  = Max( x23, x01 );
	if ( ascending ) { r[0]=r0; r[1]=r1; r[2]=r2; r[3]=r3; }
	else             { r[0]=r3; r[1]=r2; r[2]=r1; r[3]=r0; }
}

//////////////////////////////////////////////////////////////////////////

//-------------------------------------------------------------------------------
} // namespace cy
//-------------------------------------------------------------------------------

