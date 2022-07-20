#----------------------------------------------------------------
# Generated CMake target import file.
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "stdgpu::stdgpu" for configuration ""
set_property(TARGET stdgpu::stdgpu APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(stdgpu::stdgpu PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_NOCONFIG "CXX"
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libstdgpu.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS stdgpu::stdgpu )
list(APPEND _IMPORT_CHECK_FILES_FOR_stdgpu::stdgpu "${_IMPORT_PREFIX}/lib/libstdgpu.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
