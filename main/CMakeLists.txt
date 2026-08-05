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
  ${g_ext_MAIN_DIR}/simple.cu
)

# Output-degree ablation executable.
#
# This file contains host C++ only. CUDA kernels remain inside
# the existing g_ext library, so this source can remain .cpp.
add_executable(
  output_degree_ablation
  ${g_ext_MAIN_DIR}/output_degree_ablation.cpp
)


find_package(Threads REQUIRED)


list(
  APPEND
  g_ext_examples
  simple
  output_degree_ablation
)


# -----------------------------------------------------------------------------
# Path and library settings
# -----------------------------------------------------------------------------

foreach(example IN LISTS g_ext_examples)

  target_include_directories(
    ${example}
    PRIVATE
    ${PROJECT_SOURCE_DIR}
  )

  # Important:
  # Use ${example}, not the previously hard-coded "simple".
  # Otherwise output_degree_ablation would not link against g_ext.
  target_link_libraries(
    ${example}
    PRIVATE
    g_ext
  )

  target_link_libraries(
    ${example}
    PRIVATE
    g_ext
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