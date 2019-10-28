cmake_minimum_required(VERSION 3.15)
project(lab2)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_PREFIX_PATH /usr/lib64/openmpi)
find_package(MPI REQUIRED)

include_directories(${MPI_INCLUDE_PATH})

add_executable(lab2 Main.cpp Data.cpp Data.h KnnAlgorithm.cpp KnnAlgorithm.h LoadCSV.cpp LoadCSV.h
        Preprocessing.cpp Preprocessing.h)

target_link_libraries(lab2 ${MPI_LIBRARIES})

if(MPI_COMPILE_FLAGS)
    set_target_properties(lab2 PROPERTIES
            COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
endif()

if(MPI_LINK_FLAGS)
    set_target_properties(lab2 PROPERTIES
            LINK_FLAGS "${MPI_LINK_FLAGS}")
endif()