# File location: g-egg/unittests
enable_testing()

include(${gegg_3RD_PARTY_DIR}/doctest/doctest.cmake)
# string(FIND '${CMAKE_CXX_FLAGS}' "-fsanitize" sanitize)

find_package(Threads REQUIRED)

list(APPEND gegg_UNITTESTS 
  test_basics
)

# Add unittests for gegg
foreach(unittest IN LISTS gegg_UNITTESTS)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build/unittests)
  add_executable(${unittest} ${unittest}.cpp)

  # Set compiler flags for the "unittests" target
  target_compile_options(${unittest} PRIVATE
    -Wall
    -Wextra
    -Wfatal-errors
    -Wshadow
  )

  target_link_libraries(${unittest} 
    PRIVATE Threads::Threads
    PUBLIC gegg
  )
  
  target_include_directories(${unittest} 
    PRIVATE 
    ${gegg_3RD_PARTY_DIR}/doctest
    PUBLIC
    ${gegg_3RD_PARTY_DIR}/taskflow
    ${PROJECT_SOURCE_DIR}
  )
  doctest_discover_tests(${unittest})

endforeach()

# Add the test to CTest
# Add the unit tests to the test target
foreach(unittest IN LISTS gegg_UNITTESTS)
  add_test(NAME ${unittest} COMMAND ${unittest})
endforeach()


