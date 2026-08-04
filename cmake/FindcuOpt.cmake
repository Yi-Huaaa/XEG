# Locate the NVIDIA cuOpt C API installed by the libcuopt Python wheel.
#
# Input:
#   CUOPT_ROOT - package root containing include/ and lib64/, for example:
#     .../site-packages/libcuopt
#
# Output:
#   cuOpt_FOUND
#   CUOPT_INCLUDE_DIR
#   CUOPT_LIBRARY
#   CUOPT_RUNTIME_LIBRARY_DIRS
#   cuOpt::cuOpt imported target

if(NOT CUOPT_ROOT AND DEFINED ENV{CUOPT_ROOT})
  set(CUOPT_ROOT "$ENV{CUOPT_ROOT}")
endif()

find_path(
  CUOPT_INCLUDE_DIR
  NAMES cuopt/linear_programming/cuopt_c.h
  HINTS "${CUOPT_ROOT}"
  PATH_SUFFIXES include
  NO_DEFAULT_PATH
)

find_library(
  CUOPT_LIBRARY
  NAMES cuopt libcuopt
  HINTS "${CUOPT_ROOT}"
  PATH_SUFFIXES lib64 lib
  NO_DEFAULT_PATH
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  cuOpt
  REQUIRED_VARS CUOPT_INCLUDE_DIR CUOPT_LIBRARY
)

if(cuOpt_FOUND AND NOT TARGET cuOpt::cuOpt)
  add_library(cuOpt::cuOpt SHARED IMPORTED)
  set_target_properties(
    cuOpt::cuOpt
    PROPERTIES
      IMPORTED_LOCATION "${CUOPT_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${CUOPT_INCLUDE_DIR}"
  )
endif()

# The pip wheel stores libcuopt dependencies in sibling packages under the
# same site-packages directory. Record their library directories so targets
# can use a target-local build RPATH without changing LD_LIBRARY_PATH.
set(CUOPT_RUNTIME_LIBRARY_DIRS "")

if(cuOpt_FOUND)
  get_filename_component(_cuopt_site_packages "${CUOPT_ROOT}" DIRECTORY)

  list(APPEND CUOPT_RUNTIME_LIBRARY_DIRS "${CUOPT_ROOT}/lib64")

  file(
    GLOB _cuopt_sibling_library_dirs
    LIST_DIRECTORIES true
    "${_cuopt_site_packages}/*/lib"
    "${_cuopt_site_packages}/*/lib64"
  )

  file(
    GLOB _cuopt_nvidia_library_dirs
    LIST_DIRECTORIES true
    "${_cuopt_site_packages}/nvidia/*/lib"
    "${_cuopt_site_packages}/nvidia/*/lib64"
  )

  list(
    APPEND CUOPT_RUNTIME_LIBRARY_DIRS
    ${_cuopt_sibling_library_dirs}
    ${_cuopt_nvidia_library_dirs}
  )
  list(REMOVE_DUPLICATES CUOPT_RUNTIME_LIBRARY_DIRS)
endif()

mark_as_advanced(CUOPT_INCLUDE_DIR CUOPT_LIBRARY)
