# File location: gpu_extraction/main
# CMakeLists.txt in the main directory

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build/main)

add_executable(simple 
    ${g_ext_MAIN_DIR}/simple.cu
)


find_package(Threads REQUIRED)

list(APPEND g_ext_examples
  simple
)

# -----------------------------------------------------------------------------
# OR-Tools: 假設在 $HOME/or-tools_x86_64_Ubuntu-22.04_cpp_v9.7.2996
# Note: my OR-Tools locates at ~/
# -----------------------------------------------------------------------------
set(ORTOOLS_HOME "$ENV{HOME}/or-tools_x86_64_Ubuntu-22.04_cpp_v9.7.2996")
include_directories(${ORTOOLS_HOME}/include)
link_directories(${ORTOOLS_HOME}/lib)

# -----------------------------------------------------------------------------
# Path Settings
# -----------------------------------------------------------------------------
foreach(example IN LISTS g_ext_examples) 
  # Link g_ext libraries
  target_include_directories(${example} PRIVATE ${PROJECT_SOURCE_DIR})
  target_link_libraries(simple PRIVATE g_ext)

  # Link g_ext libraries
  # target_link_libraries(${example} PUBLIC Threads::Threads PRIVATE g_ext)

  # Link cuBLAS library
  target_link_libraries(${example} PRIVATE ${CUBLAS_LIBRARIES})

  # Include cuBLAS include directories
  target_include_directories(${example} PRIVATE ${CUBLAS_INCLUDE_DIRS})

  # OR-Tools + dependencies
  target_link_libraries(${example}
  PRIVATE
    ${ORTOOLS_HOME}/lib/libortools.so
    protobuf
    pthread
  )
endforeach()


# -----------------------------------------------------------------------------
# Set compiler flags for the "main" target
# -----------------------------------------------------------------------------
target_compile_options(simple PRIVATE
  -Wall
  -Wextra
  -Wfatal-errors
  -Wshadow
  -O3
  -lineinfo
)


