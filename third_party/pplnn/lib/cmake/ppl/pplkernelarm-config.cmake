cmake_minimum_required(VERSION 3.10)

if(TARGET "pplkernelarm_static")
    return()
endif()

add_library(pplkernelarm_static STATIC IMPORTED)

if(NOT TARGET "pplcommon_static")
    include(${CMAKE_CURRENT_LIST_DIR}/pplcommon-config.cmake)
endif()

get_filename_component(__PPLNN_ARM_LIB_PATH__ "${CMAKE_CURRENT_LIST_DIR}/../../../libpplkernelarm_static.a" ABSOLUTE)
set_target_properties(pplkernelarm_static PROPERTIES
    INTERFACE_LINK_LIBRARIES "pplcommon_static"
    IMPORTED_LOCATION "${__PPLNN_ARM_LIB_PATH__}"
    IMPORTED_LOCATION_DEBUG "${__PPLNN_ARM_LIB_PATH__}"
    IMPORTED_LOCATION_RELEASE "${__PPLNN_ARM_LIB_PATH__}")
unset(__PPLNN_ARM_LIB_PATH__)
