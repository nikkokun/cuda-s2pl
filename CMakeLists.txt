cmake_minimum_required(VERSION 3.17)
project(cuda_s2pl CUDA)

set(CMAKE_CUDA_STANDARD 11)

add_executable(cuda_s2pl main.cu)

set_target_properties(
        cuda_s2pl
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)