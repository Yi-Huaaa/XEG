# File location: gpu_extraction/unittests
enable_testing()

include(${g_ext_3RD_PARTY_DIR}/doctest/doctest.cmake)
# string(FIND '${CMAKE_CXX_FLAGS}' "-fsanitize" sanitize)

find_package(Threads REQUIRED)

list(APPEND g_ext_UNITTESTS
  test_basics
  test_cpu_extraction
  test_gpu_tree
  test_ilp_backend
  test_xeg_milp_model
  test_csr_invariants
  test_gpu_dag_construction
  test_gpu_finalize
  test_gpu_batch_insertion
  test_gpu_direct_insertion
  test_gpu_warp_insertion
  test_gpu_warp_bitmap_insertion
  test_gpu_block_insertion
  test_gpu_block_bitmap_insertion
  test_gpu_batch_then_direct
  test_gpu_levelization
)

# Add unittests for g_ext
foreach(unittest IN LISTS g_ext_UNITTESTS)
  set(
    CMAKE_RUNTIME_OUTPUT_DIRECTORY
    ${CMAKE_BINARY_DIR}/unittests
  )

  add_executable(
    ${unittest}
    ${unittest}.cpp
  )

  target_compile_definitions(
    ${unittest}
    PRIVATE
      G_EXT_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
  )

  # Set compiler flags for the "unittests" target
  target_compile_options(${unittest} PRIVATE
    -Wall
    -Wextra
    -Wfatal-errors
    -Wshadow
  )

  target_link_libraries(${unittest}
    PRIVATE Threads::Threads
    PUBLIC g_ext
  )

  target_include_directories(${unittest}
    PRIVATE
    ${g_ext_3RD_PARTY_DIR}/doctest
    PUBLIC
    ${g_ext_3RD_PARTY_DIR}/taskflow
    ${PROJECT_SOURCE_DIR}
  )

  doctest_discover_tests(${unittest})
endforeach()

# Add the test to CTest
# Add the unit tests to the test target
foreach(unittest IN LISTS g_ext_UNITTESTS)
  add_test(NAME ${unittest} COMMAND ${unittest})
endforeach()


# -----------------------------------------------------------------------------
# Standalone fixed-mask comparison test for GPU DAG levelization
#
# This test has its own main() and directly compares the host-driven and
# hybrid implementations. It is intentionally kept outside g_ext_UNITTESTS
# because that list assumes .cpp doctest executables.
# -----------------------------------------------------------------------------

set(
  GPU_DAG_HYBRID_FIXED_MASK_TEST
  test_gpu_dag_hybrid_fixed_mask
)

add_executable(
  ${GPU_DAG_HYBRID_FIXED_MASK_TEST}
  test_gpu_dag_hybrid_fixed_mask.cu
)

# Keep the executable beside the existing unit-test executables.
#
# Explicitly retain separable CUDA compilation because this executable
# links against the CUDA g_ext library.
set_target_properties(
  ${GPU_DAG_HYBRID_FIXED_MASK_TEST}
  PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY
      "${CMAKE_BINARY_DIR}/unittests"
    CUDA_SEPARABLE_COMPILATION
      ON
)

# Match the dependencies used by the existing GPU unit tests.
target_link_libraries(
  ${GPU_DAG_HYBRID_FIXED_MASK_TEST}
  PRIVATE
    Threads::Threads
    g_ext
)

# Required for:
#
#   #include <g_ext/G_EXT.hpp>
target_include_directories(
  ${GPU_DAG_HYBRID_FIXED_MASK_TEST}
  PRIVATE
    ${PROJECT_SOURCE_DIR}
)

# This executable uses a custom main() rather than doctest, so register it
# directly with CTest instead of calling doctest_discover_tests().
add_test(
  NAME
    ${GPU_DAG_HYBRID_FIXED_MASK_TEST}
  COMMAND
    $<TARGET_FILE:${GPU_DAG_HYBRID_FIXED_MASK_TEST}>
)

# -----------------------------------------------------------------------------
# Optional cuOpt C API smoke test
# -----------------------------------------------------------------------------
if(XEG_ENABLE_CUOPT)
  add_executable(
    test_cuopt_link
    test_cuopt_link.cpp
  )

  set_target_properties(
    test_cuopt_link
    PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/unittests"
      BUILD_RPATH "${CUOPT_RUNTIME_LIBRARY_DIRS}"
  )

  target_compile_options(
    test_cuopt_link
    PRIVATE
      -Wall
      -Wextra
      -Wfatal-errors
      -Wshadow
  )

  target_include_directories(
    test_cuopt_link
    PRIVATE
      ${g_ext_3RD_PARTY_DIR}/doctest
  )

  target_link_libraries(
    test_cuopt_link
    PRIVATE
      cuOpt::cuOpt
  )

  doctest_discover_tests(test_cuopt_link)
endif()
