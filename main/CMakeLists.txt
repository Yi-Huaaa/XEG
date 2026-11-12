# File location: gpu_extraction/main
# CMakeLists.txt in the main directory

set(
  CMAKE_RUNTIME_OUTPUT_DIRECTORY
  ${CMAKE_BINARY_DIR}/main
)

# -----------------------------------------------------------------------------
# Executables
# -----------------------------------------------------------------------------

# Existing production/example executable.
#
# Keep this source and target unchanged.
add_executable(
  simple
  ${xeg_MAIN_DIR}/simple.cu
)

# Output-degree ablation executable.
#
# This file contains host C++ only. CUDA kernels remain inside
# the existing xeg library, so this source can remain .cpp.
add_executable(
  output_degree_ablation
  ${xeg_MAIN_DIR}/output_degree_ablation.cpp
)

# Standalone XEG-D executable.
#
# This path uses the complete post-read e-graph, runs CPU tree
# construction for the warm start, and solves with CP-SAT.
add_executable(
  xeg_d
  ${xeg_MAIN_DIR}/xeg_d.cpp
)


add_executable(cpu ${xeg_MAIN_DIR}/cpu.cpp)
add_executable(xeg_t ${xeg_MAIN_DIR}/xeg_t.cpp)

find_package(Threads REQUIRED)


list(
  APPEND
  xeg_examples
  cpu
  xeg_t
  simple
  output_degree_ablation
  xeg_d
)


# -----------------------------------------------------------------------------
# Path and library settings
# -----------------------------------------------------------------------------

foreach(example IN LISTS xeg_examples)

  target_include_directories(
    ${example}
    PRIVATE
    ${PROJECT_SOURCE_DIR}
  )

  # Important:
  # Use ${example}, not the previously hard-coded "simple".
  # Otherwise output_degree_ablation would not link against xeg.
  target_link_libraries(
    ${example}
    PRIVATE
    xeg
  )

  target_link_libraries(
    ${example}
    PRIVATE
    xeg
  )

endforeach()


# -----------------------------------------------------------------------------
# Compiler flags for the existing CUDA "simple" target
# -----------------------------------------------------------------------------

# Preserve the original simple target flags exactly.
target_compile_options(
  simple
  PRIVATE
  -Wall
  -Wextra
  -Wfatal-errors
  -Wshadow
  -O3
  -lineinfo
)


# -----------------------------------------------------------------------------
# Compiler flags for the host-only ablation executable
# -----------------------------------------------------------------------------

# output_degree_ablation.cpp is compiled by the normal C++
# compiler rather than NVCC.
#
# Do not pass -lineinfo here because -lineinfo is a CUDA/NVCC
# option and may be rejected by GCC or Clang.
target_compile_options(
  output_degree_ablation
  PRIVATE
  -Wall
  -Wextra
  -Wfatal-errors
  -Wshadow
  -O3
)

# xeg_d.cpp is also compiled by the normal C++ compiler.
target_compile_options(
  xeg_d
  PRIVATE
  -Wall
  -Wextra
  -Wfatal-errors
  -Wshadow
  -O3
)
