
add_executable(root_timing_cuda 
root_timing.cu)
# Enable separate compilation for CUDA files
set_target_properties(root_timing_cuda PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON)

target_compile_definitions(root_timing_cuda PRIVATE
    CUCCD_WORKSPACE_PATH="${CMAKE_CURRENT_SOURCE_DIR}"
    CUCCD_ASSET_PATH="${CMAKE_CURRENT_SOURCE_DIR}/data")

    
# target_compile_definitions(root_timing_cuda PRIVATE
#     USE_FLOAT)

target_compile_definitions(root_timing_cuda PRIVATE
    USE_FAST_MATH)
target_compile_options(root_timing_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)