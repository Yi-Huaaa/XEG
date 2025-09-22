# File location: g-egg/main
# CMakeLists.txt in the main directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build/main)
# add_executable(simple ${gegg_MAIN_DIR}/simple.cpp)
add_executable(simple 
    ${gegg_MAIN_DIR}/simple.cpp
)


find_package(Threads REQUIRED)

# 
list(APPEND gegg_examples
  simple
)

foreach(example IN LISTS gegg_examples) 
  # Link gegg libraries
  target_include_directories(${example} PRIVATE ${PROJECT_SOURCE_DIR})
  target_link_libraries(simple PRIVATE gegg)

  # Link gegg libraries
  # target_link_libraries(${example} PUBLIC Threads::Threads PRIVATE gegg)

  # Link cuBLAS library
  target_link_libraries(${example} PRIVATE ${CUBLAS_LIBRARIES})

  # Include cuBLAS include directories
  target_include_directories(${example} PRIVATE ${CUBLAS_INCLUDE_DIRS})
endforeach()


# Set compiler flags for the "main" target
target_compile_options(simple PRIVATE
  -Wall
  -Wextra
  -Wfatal-errors
  -Wshadow
  -O3
  -lineinfo
)


