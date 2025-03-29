cmake_minimum_required(VERSION 3.10)
project(CUDA_Challenges CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES 75)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# 获取当前目录下所有子目录
file(GLOB sub_dirs RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*")

foreach(sub_dir ${sub_dirs})
    # 检查是否是目录且不是CMake特殊目录
    if(IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${sub_dir} AND 
       NOT sub_dir STREQUAL "CMakeFiles" AND
       NOT sub_dir STREQUAL "build")
        message(STATUS "Processing directory: ${sub_dir}")
        
        # 查找该目录下的所有.cpp文件
        file(GLOB cpp_files "${sub_dir}/*.cpp" "${sub_dir}/*.cu" "${sub_dir}/*.h" "${sub_dir}/*.cuh")
        
        if(cpp_files)
            # 使用目录名作为可执行文件名
            add_executable(${sub_dir} ${cpp_files})
            
            # 包含当前目录作为头文件搜索路径
            target_include_directories(${sub_dir} PRIVATE ${sub_dir})
            
            message(STATUS "Added executable: ${sub_dir} from ${cpp_files}")
        else()
            message(STATUS "No .cpp files found in ${sub_dir}")
        endif()
    endif()
endforeach()