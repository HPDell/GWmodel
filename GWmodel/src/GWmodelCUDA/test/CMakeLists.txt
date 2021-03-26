cmake_minimum_required(VERSION 3.1)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Armadillo REQUIRED)
if(ARMADILLO_FOUND)
    add_definitions(-DARMA_USE_LAPACK -DARMA_USE_BLAS -DARMA_DONT_USE_WRAPPER)
    include_directories(${ARMADILLO_INCLUDE_DIR})

    find_package(BLAS)
    if(NOT BLAS_FOUND)
        message(SEND_ERROR "BLAS Not Found!")
    endif(NOT BLAS_FOUND)
    
    find_package(LAPACK)
    if(NOT LAPACK_FOUND)
        message(SEND_ERROR "LAPACK Not Found!")
    endif(NOT LAPACK_FOUND)
else(ARMADILLO_FOUND)
    message(FATAL_ERROR "Armadillo Not Found!")
endif(ARMADILLO_FOUND)

include_directories(${CMAKE_SOURCE_DIR})

set(SAMPLE_DATA_DIR ${CMAKE_CURRENT_SOURCE_DIR})
add_definitions(-DSAMPLE_DATA_DIR="${SAMPLE_DATA_DIR}")

set(SOURCE test.cpp zillow.cpp)
set(HEADER csv.h zillow.h)

add_executable(GWmodelCUDA_test ${SOURCE} ${HEADER})
target_link_libraries(GWmodelCUDA_test PRIVATE GWmodelCUDA64 ${ARMADILLO_LIBRARY})
set_property(TARGET GWmodelCUDA_test PROPERTY CUDA_ARCHITECTURES 35 37 52 60 61 70 75)
