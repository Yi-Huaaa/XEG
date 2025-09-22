# File location: gpu_extraction/g_ext
# CMakeLists.txt in the g_ext directory

enable_language(CUDA)

add_library(g_ext 
  G_EXT.hpp
  g_ext.cu g_ext.cuh
)


# -----------------------------------------------------------------------------
# Link OpenMP for CUDA partition (needed for OpenMP support in CUDA code)
# -----------------------------------------------------------------------------
# target_link_libraries(g_ext PRIVATE OpenMP::OpenMP_CXX)
target_compile_options(g_ext PRIVATE ${OpenMP_CXX_FLAGS})


# -----------------------------------------------------------------------------
# Add cuBLAS library if necessary
# -----------------------------------------------------------------------------
if (g_ext_BUILD_CUDA)
  target_link_libraries(g_ext PRIVATE ${CUBLAS_LIBRARIES})
  target_include_directories(g_ext PRIVATE ${CUBLAS_INCLUDE_DIRS})
endif()


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

# Set compiler flags for the "g_ext" library
target_compile_options(g_ext PRIVATE
  -Wall
  -Wextra
  -Wfatal-errors
  -Wshadow
  -O3  
  -lineinfo
  -DNDEBUG
)