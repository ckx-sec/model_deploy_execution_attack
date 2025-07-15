cmake_minimum_required(VERSION 3.10)

if(TARGET "pplcommon_static")
    return()
endif()

get_filename_component(__PPLCOMMON_PACKAGE_ROOTDIR__ "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)

# --------------------------------------------------------------------------- #

add_library(pplcommon_static STATIC IMPORTED)

get_filename_component(__PPLCOMMON_LIB_PATH__ "${__PPLCOMMON_PACKAGE_ROOTDIR__}/lib/libpplcommon_static.a" ABSOLUTE)
set_target_properties(pplcommon_static PROPERTIES
    INTERFACE_LINK_LIBRARIES "pthread"
    IMPORTED_LOCATION "${__PPLCOMMON_LIB_PATH__}"
    IMPORTED_LOCATION_DEBUG "${__PPLCOMMON_LIB_PATH__}"
    IMPORTED_LOCATION_RELEASE "${__PPLCOMMON_LIB_PATH__}")
unset(__PPLCOMMON_LIB_PATH__)

# --------------------------------------------------------------------------- #

# exported definitions
set(PPLCOMMON_INCLUDE_DIRS "${__PPLCOMMON_PACKAGE_ROOTDIR__}/include")
set(PPLCOMMON_LIBRARIES "pplcommon_static")

# --------------------------------------------------------------------------- #

unset(__PPLCOMMON_PACKAGE_ROOTDIR__)
