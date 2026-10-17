# File location: gpu_extraction/
# -------------------------------------------------------------------
# g_ext CMake 
#
# To enable thread sanitizer: 
# -DCMAKE_CXX_FLAGS="-fsanitize=thread -g"
#
# To enable address and leak sanitizers
# -DCMAKE_CXX_FLAGS="-fsanitize=address  -fsanitize=leak -g"
#
# To enable different c++ standards
# -DCMAKE_CXX_STANDARD=17
# -DCMAKE_CXX_STANDARD=20
# -DCMAKE_CXX_STANDARD=23
#
# To switch to different build types:
# -DCMAKE_BUILD_TYPE=DEBUG
# -DCMAKE_BUILD_TYPE=RELEASE
# -------------------------------------------------------------------

# CMake version
cmake_minimum_required(VERSION 3.20)

# CUDA architecture. Must be set before project(), which otherwise
# fills in the compiler's default. Override with
# -DCMAKE_CUDA_ARCHITECTURES=<arch>; by default build for the first
# local GPU, falling back to 86 when no GPU can be queried.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  execute_process(
    COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader
    OUTPUT_VARIABLE XEG_GPU_COMPUTE_CAP
    RESULT_VARIABLE XEG_NVIDIA_SMI_RESULT
    ERROR_QUIET
  )
  string(REGEX MATCH "[0-9]+\\.[0-9]+" XEG_GPU_COMPUTE_CAP "${XEG_GPU_COMPUTE_CAP}")
  if(XEG_NVIDIA_SMI_RESULT EQUAL 0 AND XEG_GPU_COMPUTE_CAP)
    string(REPLACE "." "" CMAKE_CUDA_ARCHITECTURES "${XEG_GPU_COMPUTE_CAP}")
  else()
    set(CMAKE_CUDA_ARCHITECTURES 86)
  endif()
endif()
message(STATUS "CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")

project(
  g_ext 
  VERSION 1.0 
  LANGUAGES C CXX CUDA
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_FLAGS "-std=c++17")
# compile and link CUDA code separately from the rest of your code to
# speed up the build process and reduce the size of your final executable.
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wall -O3 --extended-lambda")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wall -O3 --extended-lambda")


# set compilation flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to Release ")
  set(
    CMAKE_BUILD_TYPE Release
    CACHE
    STRING "Choose the type of build."
    FORCE
  )
  # Set the possible values of build type for cmake-gui
  set_property(
    CACHE
    CMAKE_BUILD_TYPE
    PROPERTY STRINGS
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo"
  )
endif()


# build options
option(g_ext_BUILD_TESTS "Enables builds of tests" ON)
option(g_ext_BUILD_CUDA "Enables builds of cuda code" ON)
option(XEG_ENABLE_CUOPT "Build the NVIDIA cuOpt backend" OFF)
set(CUOPT_ROOT "" CACHE PATH "Root of the libcuopt package")

# project-specific variables
set(g_ext_CMAKE_DIR ${PROJECT_SOURCE_DIR}/cmake)
set(g_ext_3RD_PARTY_DIR ${PROJECT_SOURCE_DIR}/3rd-party)
set(g_ext_MAIN_DIR ${PROJECT_SOURCE_DIR}/main)
message(STATUS "g_ext_3RD_PARTY_DIR: " ${g_ext_3RD_PARTY_DIR})
message(STATUS "CMAKE_CUDA_COMPILER: " ${CMAKE_CUDA_COMPILER})
message(STATUS "CMAKE_CUDA_COMPILER_ID: " ${CMAKE_CUDA_COMPILER_ID})
message(STATUS "CMAKE_CUDA_COMPILER_VERSION: " ${CMAKE_CUDA_COMPILER_VERSION})
message(STATUS "CMAKE_CUDA_FLAGS: " ${CMAKE_CUDA_FLAGS})
message(STATUS "g_ext_BUILD_CUDA: " ${g_ext_BUILD_CUDA})
message(STATUS "XEG_ENABLE_CUOPT: " ${XEG_ENABLE_CUOPT})

# Turn on the verbose
set(CMAKE_VERBOSE_MAKEFILE ON)


# -----------------------------------------------------------------------------
# must-have package include
# -----------------------------------------------------------------------------
include(CheckLanguage)

# Enable test
include(CTest)
set(CMAKE_CTEST_ARGUMENTS "--output-on-failure")

if(XEG_ENABLE_CUOPT)
  if(POLICY CMP0144)
    cmake_policy(SET CMP0144 NEW)
  endif()

  list(APPEND CMAKE_MODULE_PATH "${g_ext_CMAKE_DIR}")
  find_package(cuOpt REQUIRED)
  message(STATUS "CUOPT_INCLUDE_DIR: ${CUOPT_INCLUDE_DIR}")
  message(STATUS "CUOPT_LIBRARY: ${CUOPT_LIBRARY}")
endif()

# -----------------------------------------------------------------------------
# start build
# -----------------------------------------------------------------------------
find_package(OpenMP REQUIRED)
add_subdirectory(g_ext)
add_subdirectory(main)
add_subdirectory(unittests)


# -----------------------------------------------------------------------------
# Add OpenMP
# -----------------------------------------------------------------------------
find_package(OpenMP)

if(OpenMP_CXX_FOUND)
    message(STATUS "Found OpenMP via package: ${OpenMP_CXX_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else()
    message(WARNING "OpenMP package not found, fallback to manual flag")
    set(OpenMP_CXX_FLAGS "-fopenmp")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# find_package(OpenMP REQUIRED)
# if (OPENMP_FOUND)
#     # Add OpenMP flags to the C++ and CUDA compilers
#     set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
#     set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#     set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
#     # make the .cu files can utilize the OMP
#     set (CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=${OpenMP_CXX_FLAGS}")
# endif()




# -----------------------------------------------------------------------------
# find nvcc
# https://cliutils.gitlab.io/modern-cmake/chapters/packages/CUDA.html
# -----------------------------------------------------------------------------
if(g_ext_BUILD_CUDA)
  message(STATUS "Configuring CUDA ...")
  enable_language(CUDA)
  if(NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR "\nNo CUDA compiler found")
  endif()
  enable_language(CUDA)
  if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "11") # need 12.2 (or higher)
    message(STATUS "CMAKE_CUDA_COMPILER_VERSION: ${CMAKE_CUDA_COMPILER_VERSION}")
    message(FATAL_ERROR "\ng_ext requires CUDA at least v12 (Driver: version 550 or higher)")
  endif()

  # configure cuBLAS
  message(STATUS "Configuring CUDA BLAS ...")
  list(APPEND CMAKE_MODULE_PATH ${g_ext_CMAKE_DIR}/cuda_modules/)
  find_package(cuBLAS)
  message(STATUS "CUBLAS_FOUND: ${CUBLAS_FOUND}")
  message(STATUS "CUBLAS_INCLUDE_DIRS: ${CUBLAS_INCLUDE_DIRS}")
  message(STATUS "CUBLAS_LIBRARIES: ${CUBLAS_LIBRARIES}")

  # Set the desired CUDA compute capability
  set(CUDA_COMPUTE_CAPABILITY "86" CACHE STRING "CUDA compute capability")
endif()


# # Add cuBLAS library if necessary
# if (g_ext_BUILD_CUDA)
#     target_link_libraries(${PROJECT_NAME} INTERFACE ${CUBLAS_LIBRARIES})
#     target_include_directories(${PROJECT_NAME} INTERFACE ${CUBLAS_INCLUDE_DIRS})
# endif()

# # Add other settings
# add_library(${PROJECT_NAME} INTERFACE)
# target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_17)
# target_link_libraries(${PROJECT_NAME} INTERFACE Threads::Threads)
# target_include_directories(${PROJECT_NAME} INTERFACE
#   $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
#   $<INSTALL_INTERFACE:include/> 
# )

# include the default directory
include_directories(${PROJECT_SOURCE_DIR})

add_library(default_settings INTERFACE)
add_library(g_ext::default_settings ALIAS default_settings)


# -----------------------------------------------------------------------------
# g_ext::default_setting
# -----------------------------------------------------------------------------

# error setting
add_library(error_settings INTERFACE)

# cuda compiler options
if(g_ext_BUILD_CUDA)
  message(STATUS "Configuring CUDA compiler options ...")
  target_compile_options(
    error_settings
    BEFORE
    INTERFACE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda -Xcompiler=-Wall,-Wextra,-Wfatal-errors,-Wshadow>
  )
endif(g_ext_BUILD_CUDA)

