# File location: gpu_extraction/unittests
enable_testing()

include(${g_ext_3RD_PARTY_DIR}/doctest/doctest.cmake)
# string(FIND '${CMAKE_CXX_FLAGS}' "-fsanitize" sanitize)

find_package(Threads REQUIRED)

list(APPEND g_ext_UNITTESTS
  test_basics
  test_gpu_levelization
)

# Add unittests for g_ext
foreach(unittest IN LISTS g_ext_UNITTESTS)
  set(
    CMAKE_RUNTIME_OUTPUT_DIRECTORY
    ${PROJECT_SOURCE_DIR}/build/unittests
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


