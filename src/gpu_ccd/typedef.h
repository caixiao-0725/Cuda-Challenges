#pragma once
#ifndef TYPE_DEF_H
#define TYPE_DEF_H


    #ifdef __CUDACC__
        #define CUDA_CALLABLE __host__ __device__
        #define DEVICE_CALLABLE __device__
        #define HOST_CALLABLE __host__
        #define DEVICE_INLINE_CALLABLE __device__ __forceinline__
        #define HOST_INLINE_CALLABLE __host__ __forceinline__
        #define CUDA_INLINE_CALLABLE __host__ __device__ __forceinline__
        #define INLINE_CALLABLE __forceinline__
    #else
        #define CUDA_CALLABLE
        #define DEVICE_CALLABLE
        #define HOST_CALLABLE
        #define DEVICE_INLINE_CALLABLE
        #define CUDA_INLINE_CALLABLE inline
        #define HOST_INLINE_CALLABLE inline
        #define INLINE_CALLABLE inline
    #endif
#include <string>

namespace cuccd {

    static const std::string get_asset_path() {
        #ifdef CUCCD_ASSET_PATH
            return std::string{CUCCD_ASSET_PATH};
        #else
            return std::string{""};
        #endif
    }

    static const std::string get_workspace_path() {
        #ifdef CUCCD_WORKSPACE_PATH
            return std::string{CUCCD_WORKSPACE_PATH};
        #else
            return std::string{""};
        #endif
    }
}

#endif // TYPE_DEF_H
