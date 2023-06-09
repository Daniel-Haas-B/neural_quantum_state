cmake_minimum_required(VERSION 3.9)

project(project_2_fys4411)

# Set CXX standard to C++14
set(CMAKE_CXX_STANDARD 17)

# Check if the user has specified a build type via the command line, e.g., (in the
# build-directory)
#
# $cmake .. -DCMAKE_BUILD_TYPE=Release
#
# if not, set Debug as the default build type.
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)
endif()

# set(CMAKE_BUILD_TYPE Debug)

# Add all compiler warnings. Remember, a warning in C++ can very often lead to a bug
# later on so _fix your warnings!_
set(CMAKE_CXX_FLAGS "-Wall -Wextra")

# Add debugging flag for tools such as gdb and valgrind.
set(CMAKE_CXX_FLAGS_DEBUG "-g")

# Add full compiler optimization when in the release build type. Other options include
# -O0, -O1, -O2.
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I$(brew --prefix libomp)/include")
# set(OpenMP_CXX_LIB_NAMES "omp")

# Add directories containing header-files
include_directories(
  .
  Hamiltonians
  InitialStates
  Math
  WaveFunctions
  Solvers
  Optimizers
)

# Add sources from all directories
file(
  GLOB LIB_SOURCES "particle.cpp"
  GLOB LIB_SOURCES "sampler.cpp"
  GLOB LIB_SOURCES "system.cpp"
  GLOB LIB_SOURCES "Hamiltonians/*.cpp"
  GLOB LIB_SOURCES "InitialStates/*.cpp"
  GLOB LIB_SOURCES "Math/*.cpp"
  GLOB LIB_SOURCES "Math/pcg-cpp-0.98/include/*.hpp"
  GLOB LIB_SOURCES "Solvers/*.cpp"
  GLOB LIB_SOURCES "WaveFunctions/*.cpp"
  GLOB LIB_SOURCES "Optimizers/*.cpp"
)

# Add libary to compile path
add_library(Vmclib ${LIB_SOURCES})

# find openmp
find_package(OpenMP)
include(FindOpenMP)

if(OPENMP_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

if(NOT OpenMP_CXX_FOUND)
  message(FATAL_ERROR "OpenMP not found")
endif()

# find eigen
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# Create different executables
add_executable(nqs main.cpp)
target_link_libraries(nqs PRIVATE Vmclib)

# add_executable(tests tests.cpp)
# target_link_libraries(tests PRIVATE Vmclib)
if(OpenMP_CXX_FOUND)
  target_link_libraries(nqs PUBLIC OpenMP::OpenMP_CXX)
endif()
