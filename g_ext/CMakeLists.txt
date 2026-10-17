# File location: gpu_extraction/g_ext
# CMakeLists.txt in the g_ext directory

enable_language(CUDA)

add_library(g_ext
  G_EXT.hpp
  g_ext.cu
  g_ext.cuh
  cuopt_solver.cpp
  cuopt_solver.hpp
  xeg_milp_model.cpp
  xeg_milp_model.hpp
)

# Sources compiled by the host C++ compiler (for example main/xeg_d.cpp and
# the unit tests) include g_ext.cuh, which needs the CUDA headers.
target_include_directories(g_ext
  PUBLIC
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

# -----------------------------------------------------------------------------
# Optional NVIDIA cuOpt backend
# -----------------------------------------------------------------------------
if(XEG_ENABLE_CUOPT)
  target_compile_definitions(
    g_ext
    PUBLIC
      XEG_ENABLE_CUOPT=1
  )

  target_link_libraries(
    g_ext
    PRIVATE
      cuOpt::cuOpt
  )

  set_target_properties(
    g_ext
    PROPERTIES
      BUILD_RPATH "${CUOPT_RUNTIME_LIBRARY_DIRS}"
  )
endif()

# -----------------------------------------------------------------------------
# Link OpenMP for CUDA partition (needed for OpenMP support in CUDA code)
# -----------------------------------------------------------------------------
# target_link_libraries(g_ext PRIVATE OpenMP::OpenMP_CXX)
target_compile_options(g_ext PRIVATE ${OpenMP_CXX_FLAGS})

# -----------------------------------------------------------------------------
# Include taskflow
# -----------------------------------------------------------------------------
target_include_directories(g_ext PUBLIC ${g_ext_3RD_PARTY_DIR}/taskflow)

# -----------------------------------------------------------------------------
# Find the pthreads library
# -----------------------------------------------------------------------------
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
target_link_libraries(g_ext PRIVATE Threads::Threads)

# -----------------------------------------------------------------------------
# Find the openMP library
# -----------------------------------------------------------------------------
find_package(OpenMP REQUIRED)
target_link_libraries(g_ext PRIVATE OpenMP::OpenMP_CXX)

# -----------------------------------------------------------------------------
# OR-Tools: 假設在 $HOME/or-tools_x86_64_Ubuntu-22.04_cpp_v9.7.2996
# Note: my OR-Tools locates at ~/
# -----------------------------------------------------------------------------
# Override with -DORTOOLS_HOME=<dir> or the ORTOOLS_HOME environment variable.
if(DEFINED ENV{ORTOOLS_HOME})
  set(XEG_DEFAULT_ORTOOLS_HOME "$ENV{ORTOOLS_HOME}")
else()
  set(XEG_DEFAULT_ORTOOLS_HOME "$ENV{HOME}/or-tools_x86_64_Ubuntu-22.04_cpp_v9.7.2996")
endif()
set(ORTOOLS_HOME "${XEG_DEFAULT_ORTOOLS_HOME}" CACHE PATH "OR-Tools C++ install directory")
message(STATUS "ORTOOLS_HOME: ${ORTOOLS_HOME}")
target_include_directories(g_ext
  PUBLIC
    ${ORTOOLS_HOME}/include
)
target_link_directories(g_ext
  PUBLIC
    ${ORTOOLS_HOME}/lib
)
target_link_libraries(g_ext
  PRIVATE
    ${ORTOOLS_HOME}/lib/libortools.so
    protobuf
    pthread
)

# -----------------------------------------------------------------------------
# Set compiler flags for the "g_ext" library
# -----------------------------------------------------------------------------
target_compile_options(g_ext PRIVATE
  -Wall
  -Wextra
  -Wfatal-errors
  -Wshadow
  -O3  
  -lineinfo
  -DNDEBUG
)